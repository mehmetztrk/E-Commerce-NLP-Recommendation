import React, { useEffect, useMemo, useRef, useState } from 'react'
import data from './data/products.json'
import { buildTfIdf, cosine, parseQuery, vectorizeQuery } from './lib/nlp'

type Product = {
  id: string
  name: string
  price: number
  category: string
  rating: number
  description: string
}

const PRODUCTS: Product[] = data as Product[]
const CATEGORIES = Array.from(new Set(PRODUCTS.map(p => p.category)))
type SortBy = 'relevance' | 'price-asc' | 'price-desc' | 'rating-desc'

// --- tiny debounce hook (stabilize quick submits) ---
function useDebouncedValue<T>(value: T, delay = 120) {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), delay)
    return () => clearTimeout(t)
  }, [value, delay])
  return v
}

// --- simple fuzzy match helpers (typo tolerance) ---
function editDistance(a: string, b: string) {
  a = a.toLowerCase(); b = b.toLowerCase()
  const dp = Array.from({ length: a.length + 1 }, () => Array(b.length + 1).fill(0))
  for (let i = 0; i <= a.length; i++) dp[i][0] = i
  for (let j = 0; j <= b.length; j++) dp[0][j] = j
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,      // deletion
        dp[i][j - 1] + 1,      // insertion
        dp[i - 1][j - 1] + cost // substitution
      )
    }
  }
  return dp[a.length][b.length]
}
function fuzzyIncludes(hayTokens: string[], kw: string) {
  const k = kw.toLowerCase()
  const thr = k.length <= 4 ? 1 : 2 // shorter words need stricter threshold
  if (hayTokens.some(t => t.includes(k) || k.includes(t))) return true
  return hayTokens.some(t => editDistance(t, k) <= thr)
}

export default function App() {
  // typing state (doesn't trigger search)
  const [query, setQuery] = useState('')

  // only submittedQuery drives NLP & results
  const [submittedQuery, setSubmittedQuery] = useState('')

  // filters
  const [category, setCategory] = useState<string>('All')
  const [priceMin, setPriceMin] = useState<string>('')
  const [priceMax, setPriceMax] = useState<string>('')
  const [ratingMin, setRatingMin] = useState<string>('')

  // touched tracker: if user changed a field, AI must not overwrite it
  const touched = useRef({ category:false, priceMin:false, priceMax:false, ratingMin:false })

  // remember what AI auto-applied last time (to revert when user clears)
  const lastAuto = useRef<{category?:string, priceMin?:string, priceMax?:string, ratingMin?:string}>({})

  const [sortBy, setSortBy] = useState<SortBy>('relevance')

  // build TF-IDF once
  const model = useMemo(() => {
    const docs = PRODUCTS.map(p => ({ id: p.id, text: `${p.name} ${p.description} ${p.category}` }))
    return buildTfIdf(docs)
  }, [])

  const debouncedSubmitted = useDebouncedValue(submittedQuery, 100)

  const results = useMemo(() => {
    const q = debouncedSubmitted
    const parsed = parseQuery(q)
    const qVec = vectorizeQuery(q, model)

    // manual filters
    const base = PRODUCTS
      .filter(p => (category === 'All' ? true : p.category === category))
      .filter(p => (priceMin ? p.price >= parseFloat(priceMin) : true))
      .filter(p => (priceMax ? p.price <= parseFloat(priceMax) : true))
      .filter(p => (ratingMin ? p.rating >= parseFloat(ratingMin) : true))

    // parsed constraints
    const byParsed = base
      .filter(p => (parsed.category ? p.category === parsed.category : true))
      .filter(p => (parsed.priceMin !== undefined ? p.price >= (parsed.priceMin as number) : true))
      .filter(p => (parsed.priceMax !== undefined ? p.price <= (parsed.priceMax as number) : true))
      .filter(p => (parsed.ratingMin !== undefined ? p.rating >= (parsed.ratingMin as number) : true))

    // tf-idf scoring (+ category boost)
    const scoredInitial = byParsed.map(p => {
      const docVec = model.docVectors.get(p.id)!
      const rel = cosine(qVec, docVec)
      const boost = (parsed.category && p.category === parsed.category) ? 0.25 : 0
      return { product: p, score: rel + boost }
    })

    // Fallback: if nearly zero → fuzzy keyword filter
    let list = scoredInitial
    const allZero = list.length > 0 && list.every(x => x.score < 0.001)
    if (allZero) {
      const kws = (parsed.keywords || []).slice(0, 5)
      if (kws.length) {
        list = list
          .filter(x => {
            const hay = (x.product.name + ' ' + x.product.description + ' ' + x.product.category).toLowerCase()
            const hayTokens = hay.split(/[^a-z0-9\-]+/).filter(Boolean)
            return kws.every(k => fuzzyIncludes(hayTokens, k))
          })
          .map(x => {
            const catBoost = (parsed.category && x.product.category === parsed.category) ? 0.4 : 0
            return { ...x, score: x.score + 0.25 + catBoost }
          })
      }
      if (list.length === 0) {
        list = byParsed.map(p => ({ product: p, score: 0.12 }))
      }
    }

    // sorting
    if (sortBy === 'relevance')      list.sort((a,b) => b.score - a.score)
    else if (sortBy === 'price-asc') list.sort((a,b) => a.product.price - b.product.price)
    else if (sortBy === 'price-desc')list.sort((a,b) => b.product.price - a.product.price)
    else if (sortBy === 'rating-desc') list.sort((a,b) => b.product.rating - a.product.rating)

    return { scored: list, parsed }
  }, [debouncedSubmitted, category, priceMin, priceMax, ratingMin, sortBy, model])

  // apply parsed → filters only when user explicitly searches (and only for untouched fields)
  const applyParsedToFilters = (parsed: ReturnType<typeof parseQuery>) => {
    const auto: typeof lastAuto.current = {}
    if (parsed.category && !touched.current.category) {
      setCategory(parsed.category); auto.category = parsed.category
    }
    if (parsed.priceMin !== undefined && !touched.current.priceMin) {
      const v = String(parsed.priceMin); setPriceMin(v); auto.priceMin = v
    }
    if (parsed.priceMax !== undefined && !touched.current.priceMax) {
      const v = String(parsed.priceMax); setPriceMax(v); auto.priceMax = v
    }
    if (parsed.ratingMin !== undefined && !touched.current.ratingMin) {
      const v = String(parsed.ratingMin); setRatingMin(v); auto.ratingMin = v
    }
    lastAuto.current = auto
  }

  const handleSearch = () => {
    const trimmed = query.trim()
    setSubmittedQuery(trimmed)

    if (trimmed === '') {
      // user cleared → revert AI-applied fields only (if user hasn't touched them)
      const auto = lastAuto.current
      if (auto.category !== undefined && !touched.current.category) setCategory('All')
      if (auto.priceMin !== undefined && !touched.current.priceMin) setPriceMin('')
      if (auto.priceMax !== undefined && !touched.current.priceMax) setPriceMax('')
      if (auto.ratingMin !== undefined && !touched.current.ratingMin) setRatingMin('')
      lastAuto.current = {}
      return
    }
    const parsed = parseQuery(trimmed)
    applyParsedToFilters(parsed)
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') handleSearch()
  }

  const mark = (k: keyof typeof touched.current) => { touched.current[k] = true }

  return (
    <div style={{maxWidth: 1100, margin: '0 auto', padding: '24px'}}>
      <header style={{display:'flex', alignItems:'center', gap:12, justifyContent:'space-between', marginBottom: 20}}>
        <div style={{display:'flex', alignItems:'center', gap:10}}>
          <div style={{width: 36, height: 36, borderRadius: 10, background: 'linear-gradient(135deg, #5dd6ff, #8a7dff)'}}></div>
          <h1 style={{margin:0, fontSize: 22}}>E-Shop • AI Smart Search</h1>
        </div>
        <div style={{opacity:.8}}>Demo catalog • {PRODUCTS.length} products</div>
      </header>

      <section style={{display:'grid', gridTemplateColumns: '1fr', gap: 14, background:'#12161d', padding:16, borderRadius:12, boxShadow:'0 10px 30px rgba(0,0,0,0.35)'}}>
        <div style={{display:'grid', gridTemplateColumns:'1fr 110px 160px', gap: 10}}>
          <input
            value={query}
            onChange={e=>setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder='Try: "running shoes under $100 with good reviews"'
            style={{width:'100%', padding:'12px 14px', borderRadius:10, border:'1px solid #2a3342', background:'#0b0f16', color:'#eef2ff'}}
          />
          <button onClick={handleSearch}
            style={{padding:'12px 14px', borderRadius:10, border:'1px solid #2a3342', background:'linear-gradient(135deg,#5dd6ff20,#8a7dff20)', color:'#eef2ff'}}>Search</button>
          <select value={sortBy} onChange={e=>setSortBy(e.target.value as SortBy)} style={{padding:'12px 14px', borderRadius:10, border:'1px solid #2a3342', background:'#0b0f16', color:'#eef2ff'}}>
            <option value="relevance">Sort: Relevance</option>
            <option value="price-asc">Price: Low → High</option>
            <option value="price-desc">Price: High → Low</option>
            <option value="rating-desc">Rating: High → Low</option>
          </select>
        </div>

        <div style={{display:'grid', gridTemplateColumns:'repeat(5, 1fr)', gap:10}}>
          <div>
            <label style={{display:'block', fontSize:12, opacity:.8, marginBottom:4}}>Category</label>
            <select value={category} onChange={e=>{ setCategory(e.target.value); mark('category') }} style={{width:'100%', padding:'10px 12px', borderRadius:10, border:'1px solid #2a3342', background:'#0b0f16', color:'#eef2ff'}}>
              <option>All</option>
              {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div>
            <label style={{display:'block', fontSize:12, opacity:.8, marginBottom:4}}>Min Price</label>
            <input value={priceMin} onChange={e=>{ setPriceMin(e.target.value); mark('priceMin') }} placeholder='0' type='number' min='0' step='0.01' style={{width:'100%', padding:'10px 12px', borderRadius:10, border:'1px solid #2a3342', background:'#0b0f16', color:'#eef2ff'}}/>
          </div>
          <div>
            <label style={{display:'block', fontSize:12, opacity:.8, marginBottom:4}}>Max Price</label>
            <input value={priceMax} onChange={e=>{ setPriceMax(e.target.value); mark('priceMax') }} placeholder='200' type='number' min='0' step='0.01' style={{width:'100%', padding:'10px 12px', borderRadius:10, border:'1px solid #2a3342', background:'#0b0f16', color:'#eef2ff'}}/>
          </div>
          <div>
            <label style={{display:'block', fontSize:12, opacity:.8, marginBottom:4}}>Min Rating</label>
            <input value={ratingMin} onChange={e=>{ setRatingMin(e.target.value); mark('ratingMin') }} placeholder='4.0' type='number' min='0' max='5' step='0.1' style={{width:'100%', padding:'10px 12px', borderRadius:10, border:'1px solid #2a3342', background:'#0b0f16', color:'#eef2ff'}}/>
          </div>
          <div style={{display:'flex', gap:8, alignItems:'flex-end'}}>
            <button onClick={()=>{ setQuery('running shoes under $100 with good reviews'); }}
              style={{flex:1, padding:'10px 12px', borderRadius:10, border:'1px solid #2a3342', background:'linear-gradient(135deg,#5dd6ff20,#8a7dff20)', color:'#eef2ff'}}>Example</button>
            <button onClick={()=>{
              setCategory('All'); setPriceMin(''); setPriceMax(''); setRatingMin('');
              setQuery(''); setSubmittedQuery(''); lastAuto.current = {};
              touched.current = { category:false, priceMin:false, priceMax:false, ratingMin:false }
            }}
              style={{padding:'10px 12px', borderRadius:10, border:'1px solid #2a3342', background:'#0b0f16', color:'#eef2ff'}}>Reset</button>
          </div>
        </div>

        {/* small hint: last executed search */}
        <div style={{fontSize:12, opacity:.7}}>
          Last search: <code style={{opacity:.9}}>{submittedQuery || '—'}</code>
        </div>
      </section>

      {/* stabilize layout height to avoid jumps */}
      <div style={{minHeight: 320, marginTop: 16}}>
        <ResultsGrid results={results.scored} />
      </div>

      <footer style={{opacity:.7, marginTop:20, fontSize:12}}>
       Mehmet Ozturk.
      </footer>
    </div>
  )
}

function ResultsGrid({results}:{results:{product: Product, score: number}[]}){
  return (
    <div style={{display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(260px, 1fr))', gap:16}}>
      {results.map(({product, score}) => (
        <div key={product.id} style={{background:'#12161d', padding:14, borderRadius:12, border:'1px solid #202633'}}>
          <div style={{display:'flex', justifyContent:'space-between', alignItems:'baseline', marginBottom:6}}>
            <h3 style={{margin:'6px 0 2px', fontSize:18}}>{product.name}</h3>
            <span style={{fontWeight:700}}>${product.price.toFixed(2)}</span>
          </div>
          <div style={{display:'flex', gap:8, alignItems:'center', fontSize:12, opacity:.85}}>
            <span style={{padding:'4px 8px', border:'1px solid #2a3342', borderRadius:999}}>{product.category}</span>
            <span>★ {product.rating.toFixed(1)}</span>
            <span style={{opacity:.6}}>relevance: {score.toFixed(2)}</span>
          </div>
          <p style={{opacity:.9}}>{product.description}</p>
          <button style={{marginTop:8, width:'100%', padding:'10px 12px', borderRadius:10, border:'1px solid #2a3342', background:'linear-gradient(135deg,#5dd6ff20,#8a7dff20)', color:'#eef2ff'}}>Add to Cart</button>
        </div>
      ))}
      {results.length === 0 && (
        <div style={{opacity:.8}}>No products found. Try relaxing filters or changing the query.</div>
      )}
    </div>
  )
}
