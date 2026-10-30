export type Gender = 'male' | 'female';
 
export interface UserProfile {
  id: string;
  nickname: string | null;
  gender: Gender | null;
  birth_year: number | null;
  fitness_goal: string | null;
  // 가족력: 부·모·형제자매 중 진단자 여부 (null = 미응답)
  fh_hypertension: boolean | null;
  fh_diabetes: boolean | null;
  fh_dyslipidemia: boolean | null;
  created_at: string;
}
 
export interface UserUpdate {
  nickname?: string | null;
  gender?: Gender | null;
  birth_year?: number | null;
  fitness_goal?: string | null;
  fh_hypertension?: boolean | null;
  fh_diabetes?: boolean | null;
  fh_dyslipidemia?: boolean | null;
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
  source_ids: string[];
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
  source_ids: string[];
}
 
export interface LifestylePlanItem {
  category: 'nutrition' | 'exercise' | 'sleep' | 'habit';
  title: string;
  reason: string;
  actions: string[];
  source_ids: string[];
}
 
export interface SupplementCandidate {
  name: string;
  purpose: string;
  related_metrics: string[];
  evidence_level: 'general' | 'moderate' | 'limited';
  why_consider: string;
  cautions: string[];
  not_recommended_if: string[];
  source_ids: string[];
}
 
export interface ReportCautions {
  medical_disclaimer: string;
  consult_doctor_if: string[];
}
 
// 리포트가 인용한 근거 문서. 항목들의 source_ids가 sid를 가리킨다.
export interface SourceRef {
  sid: string;
  title: string;
  publisher?: string | null;
  year?: number | null;
  page?: string | null;
  heading_path?: string | null;
  url?: string | null;
}
 
export interface FeatureTrend {
  feature_kor: string;
  dates: string[];
  values: number[];
  direction: 'up' | 'down' | 'stable';
  total_change: number;
  slope_per_90_days: number;
  is_shap_important: boolean;
  shap_directions: string[];
  is_risk_worsening: boolean;
}
 
export interface TrendAnalysis {
  record_count: number;
  span_days: number;
  feature_trends: FeatureTrend[];
}
 
export interface AIReport {
  summary: ReportSummary;
  body_composition_report: BodyCompositionReport;
  risk_report: RiskReportItem[];
  lifestyle_plan: LifestylePlanItem[];
  supplement_candidates: SupplementCandidate[];
  cautions: ReportCautions;
  disease_risks: DiseaseRisk[];
  sources: SourceRef[];
  trend_analysis: TrendAnalysis | null;
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
  // 2026-10-28 DB에 추가. 리포트 개인화에 쓰인다 (기존 기록은 null)
  height?: number | null; // 신장(cm)
  muscle_control?: number | null; // 결과지 "근육조절"(kg). 양수면 그만큼 늘려야 표준
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
  height?: number;
  muscle_control?: number;
  image_url?: string;
}
 
export type AnalysisResponse = AIReport;
 
 
export type InBodyUpdateInput = Partial<Omit<InBodyCreateInput, 'image_url'>>;
 
export interface InBodyUploadResponse {
  record: InBodyRecord;
  extraction_confidence: number;
  extraction_notes: string | null;
}