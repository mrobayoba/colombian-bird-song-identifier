// Types mirroring libs/contracts/contracts/models.py exactly.
// If a field changes in the Pydantic contract, change it here too.

export type JobStatus =
  | "received"
  | "preprocessing"
  | "classifying"
  | "enriching"
  | "done"
  | "failed";

export interface Candidate {
  label: string; // raw label_map value, e.g. "Turdus_fuscater"
  display_name: string; // "Turdus fuscater"
  confidence: number; // 0..1
}

export interface ClassificationResult {
  label: string;
  display_name: string;
  confidence: number;
  alternatives: Candidate[];
  n_chunks: number;
  model_version: string | null;
}

export interface ConservationStatus {
  code: string; // LC/NT/VU/EN/CR/EW/EX/DD/NE
  label_es: string;
  label_en: string;
}

export interface Taxonomy {
  kingdom?: string | null;
  phylum?: string | null;
  class?: string | null;
  order?: string | null;
  family?: string | null;
  genus?: string | null;
  species?: string | null;
}

export interface BirdCard {
  label: string;
  scientific_name: string;
  english_name: string;
  description: string;
  description_source: string;
  conservation_status: ConservationStatus;
  taxonomy: Taxonomy;
  inat_sightings: number;
  xeno_canto_recordings: number;
  image_url: string;
  info_url: string;
}

export interface EnrichmentResult {
  summary_es: string;
  behavior_notes?: string | null;
  semantic_tags: string[];
  sources: string[];
  model?: string | null;
}

export interface UnifiedResponse {
  job_id: string;
  status: JobStatus;
  identification: ClassificationResult;
  card: BirdCard | null;
  enrichment: EnrichmentResult | null;
  degraded: string[];
  created_at: string;
}
