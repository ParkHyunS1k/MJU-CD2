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

export interface ReportSummary {
  title: string;
  overall_message: string;
  key_focus: string[];
}

export interface MetricInsight {
  metric: string;
  message: string;
}

export interface BodyWatchPoint {
  metric: string;
  current_value: string;
  interpretation: string;
  priority: RiskLevel;
}

export interface BodyCompositionReport {
  strengths: MetricInsight[];
  watch_points: BodyWatchPoint[];
}

export interface RiskReportItem {
  disease: string;
  risk_level: RiskLevel;
  probability: number;
  main_factors: string[];
  message: string;
}

export interface LifestylePlanItem {
  category: 'nutrition' | 'exercise' | 'sleep' | 'habit';
  title: string;
  reason: string;
  actions: string[];
}

export interface SupplementCandidate {
  name: string;
  purpose: string;
  related_metrics: string[];
  evidence_level: 'general' | 'moderate' | 'limited';
  why_consider: string;
  cautions: string[];
  not_recommended_if: string[];
}

export interface ReportCautions {
  medical_disclaimer: string;
  consult_doctor_if: string[];
}

export interface AIReport {
  summary: ReportSummary;
  body_composition_report: BodyCompositionReport;
  risk_report: RiskReportItem[];
  lifestyle_plan: LifestylePlanItem[];
  supplement_candidates: SupplementCandidate[];
  cautions: ReportCautions;
  disease_risks: DiseaseRisk[];
  trend_analysis?: unknown | null;
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
