/**
 * @file Types for the @kotoshu/worker/semantic-merge subpath — the pure
 * merge half of the semantic layer (unit-checkable without a worker).
 */

export interface Suggestion {
  word: string
  distance: number
  confidence: number
  source: string
}

export interface SemanticNeighbor {
  word: string
  score: number
}

export const SEMANTIC_SUGGEST_K: 4

export function mergeSemanticCandidates(
  dictionary: Suggestion[],
  neighbors: SemanticNeighbor[] | null,
): Suggestion[]
