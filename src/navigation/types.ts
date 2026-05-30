import type { DiseaseRisk } from '../types/models';

export type AuthStackParamList = {
  Landing: undefined;
  Login: undefined;
  Register: undefined;
};

export type InBodyStackParamList = {
  InBodyList: undefined;
  InBodyDetail: { id: string };
  InBodyCreate: undefined;
  InBodyUpload: undefined;
  InBodyReview: { id: string; confidence: number; notes: string | null };
  AnalysisScreen: { id: string };
  RiskScreen: { id: string; risks: DiseaseRisk[] };
  ReportScreen: { id: string };
};
