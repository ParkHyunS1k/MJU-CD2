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
  RiskScreen: { id: string; disease: string }; // id: 보고 있던 기록, disease: 질환 이름(예: 고혈압)
  ReportScreen: { id: string };
};

export type SettingsStackParamList = {
  SettingsMain: undefined;
  ProfileEdit: undefined;
  Policy: { kind: 'privacy' | 'terms' };
};