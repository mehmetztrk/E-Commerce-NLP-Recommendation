export type ParsedQuery = {
  keywords: string[]
  category?: string
  priceMax?: number
  priceMin?: number
  ratingMin?: number
}

const STOPWORDS = new Set([
  'the','a','an','for','with','and','or','to','of','in','on','at','me','show','find','under','below','over','above','less','than','more','good','reviews','best','top','high','low'
])

// simple stemming (very naive)
function stem(token: string) {
  return token.replace(/(ing|ers|er|s)$/i, '')
}

const CATEGORY_SYNONYMS: Record<string,string> = {
  'shoe': 'Shoes',
  'sneaker': 'Shoes',
  'runner': 'Shoes',
  'running': 'Shoes',
  'boots': 'Shoes',
  'boot': 'Shoes',
  'shirt': 'Apparel',
  'tshirt': 'Apparel',
  't-shirt': 'Apparel',
  'apparel': 'Apparel',
  'socks': 'Apparel',
  'sock': 'Apparel',
  'bottle': 'Accessories',
  'mat': 'Accessories',
  'yoga': 'Accessories',
  'accessory': 'Accessories',
  'earbuds': 'Electronics',
  'earbud': 'Electronics',
  'headphones': 'Electronics',
  'tracker': 'Electronics',
  'watch': 'Electronics',
  'electronics': 'Electronics',
  'dumbbell': 'Equipment',
  'dumbbells': 'Equipment',
  'weights': 'Equipment',
  'equipment': 'Equipment',
}

export function parseQuery(q: string): ParsedQuery {
  const parsed: ParsedQuery = { keywords: [] }
  const lower = q.toLowerCase()

  // price rules
  const under = lower.match(/(?:under|below|less than)\s*\$?(\d+(?:\.\d+)?)/)
  if (under) parsed.priceMax = parseFloat(under[1])

  const over = lower.match(/(?:over|above|more than)\s*\$?(\d+(?:\.\d+)?)/)
  if (over) parsed.priceMin = parseFloat(over[1])

  const between = lower.match(/(?:between|from)\s*\$?(\d+(?:\.\d+)?)\s*(?:and|to)\s*\$?(\d+(?:\.\d+)?)/)
  if (between) {
    parsed.priceMin = parseFloat(between[1])
    parsed.priceMax = parseFloat(between[2])
  }

  // rating rules
  const goodRev = /(good reviews|4\+|4\s*stars|rating\s*>=?\s*4)/.test(lower)
  if (goodRev) parsed.ratingMin = 4.0

  const ratingMin = lower.match(/(?:rating|stars)\s*>=?\s*(\d(?:\.\d)?)/)
  if (ratingMin) parsed.ratingMin = parseFloat(ratingMin[1])

  // category detection
  for (const [syn, cat] of Object.entries(CATEGORY_SYNONYMS)) {
    if (lower.includes(syn)) { parsed.category = cat; break; }
  }

  // tokens
  const tokens = lower.split(/[^a-z0-9\-]+/).filter(Boolean)
  const keywords = tokens
    .filter(t => !STOPWORDS.has(t))
    .map(stem)
    .filter(t => t.length > 1)
  parsed.keywords = Array.from(new Set(keywords))
  return parsed
}

// TF-IDF utilities
export type Doc = { id: string, text: string }
export type TfIdfModel = {
  vocab: string[]
  idf: Map<string, number>
  docVectors: Map<string, Map<string, number>>
}

export function buildTfIdf(docs: Doc[]): TfIdfModel {
  const vocabSet = new Set<string>()
  const docFreq = new Map<string, number>()
  const docTokens: Map<string, string[]> = new Map()

  function tokenize(text: string) {
    return text.toLowerCase().split(/[^a-z0-9\-]+/).filter(Boolean).map(stem).filter(t => !STOPWORDS.has(t))
  }

  // collect tokens & doc frequency
  for (const d of docs) {
    const toks = tokenize(d.text)
    docTokens.set(d.id, toks)
    const seen = new Set<string>()
    for (const t of toks) {
      vocabSet.add(t)
      if (!seen.has(t)) {
        docFreq.set(t, (docFreq.get(t) || 0) + 1)
        seen.add(t)
      }
    }
  }
  const vocab = Array.from(vocabSet)
  const N = docs.length
  const idf = new Map<string, number>()
  for (const term of vocab) {
    const df = docFreq.get(term) || 0
    idf.set(term, Math.log((N + 1) / (df + 1)) + 1)
  }

  // doc vectors (tf-idf)
  const docVectors = new Map<string, Map<string, number>>()
  for (const d of docs) {
    const tf = new Map<string, number>()
    for (const t of docTokens.get(d.id) || []) tf.set(t, (tf.get(t) || 0) + 1)
    const vec = new Map<string, number>()
    for (const [t, c] of tf) {
      vec.set(t, (c / (docTokens.get(d.id)?.length || 1)) * (idf.get(t) || 0))
    }
    docVectors.set(d.id, vec)
  }

  return { vocab, idf, docVectors }
}

export function vectorizeQuery(q: string, model: TfIdfModel): Map<string, number> {
  const toks = q.toLowerCase().split(/[^a-z0-9\-]+/).filter(Boolean).map(stem).filter(t => !STOPWORDS.has(t))
  const tf = new Map<string, number>()
  for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1)
  const vec = new Map<string, number>()
  const len = toks.length || 1
  for (const [t, c] of tf) {
    vec.set(t, (c / len) * (model.idf.get(t) || 0))
  }
  return vec
}

export function cosine(a: Map<string, number>, b: Map<string, number>): number {
  let dot = 0, na = 0, nb = 0
  for (const [k, va] of a) {
    dot += va * (b.get(k) || 0)
    na += va * va
  }
  for (const [, vb] of b) nb += vb * vb
  if (na === 0 || nb === 0) return 0
  return dot / (Math.sqrt(na) * Math.sqrt(nb))
}
