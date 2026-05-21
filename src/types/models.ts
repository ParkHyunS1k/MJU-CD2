export type Gender = 'male' | 'female';

export interface UserProfile {
  id: string;
  nickname: string | null;
  gender: Gender | null;
  birth_year: number | null;
  fitness_goal: string | null;
  created_at: string;
}

export interface UserUpdate {
  nickname?: string | null;
  gender?: Gender | null;
  birth_year?: number | null;
  fitness_goal?: string | null;
}

export type RiskLevel = 'low' | 'medium' | 'high';

export interface DiseaseRisk {
  disease: string;
  probability: number;
  risk_level: RiskLevel;
  top_factors: { feature_kor: string; value: number; shap_value: number; direction: 'up' | 'down' }[];
}

export interface AIReport {
  disease_risks: DiseaseRisk[];
  lifestyle_tips: string[];
  supplement_tips: string[];
  cautions: string | null;
}

export interface InBodyRecord {
  id: string;
  user_id: string;
  measured_at: string | null;
  weight: number;
  skeletal_muscle_mass: number | null;
  body_fat_mass: number | null;
  body_fat_percentage: number | null;
  body_water: number | null;
  protein: number | null;
  minerals: number | null;
  bmi: number | null;
  basal_metabolic_rate: number | null;
  abs_fat: number | null;
  fat_level: number | null;
  image_url: string | null;
  ai_report: AIReport | null;
}

export interface InBodyCreateInput {
  measured_at?: string;
  weight: number;
  skeletal_muscle_mass?: number;
  body_fat_mass?: number;
  body_fat_percentage?: number;
  body_water?: number;
  protein?: number;
  minerals?: number;
  bmi?: number;
  basal_metabolic_rate?: number;
  abs_fat?: number;
  fat_level?: number;
  image_url?: string;
}

export type AnalysisResponse = AIReport;


export type InBodyUpdateInput = Partial<Omit<InBodyCreateInput, 'image_url'>>;

export interface InBodyUploadResponse {
  record: InBodyRecord;
  extraction_confidence: number;
  extraction_notes: string | null;
}
