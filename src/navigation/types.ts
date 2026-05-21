import type { DiseaseRisk } from '../types/models';

export type InBodyStackParamList = {
  InBodyList: undefined;
  InBodyDetail: { id: string };
  InBodyCreate: undefined;
  InBodyUpload: undefined;
  InBodyReview: { id: string; confidence: number; notes: string | null };
  RiskScreen: { id: string; risks: DiseaseRisk[] };
  ReportScreen: { id: string };
};
