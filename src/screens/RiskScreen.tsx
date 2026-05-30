import React from 'react';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import type { InBodyStackParamList } from '../navigation/types';
import type { DiseaseRisk, RiskLevel } from '../types/models';

type Nav = NativeStackNavigationProp<InBodyStackParamList, 'RiskScreen'>;
type Rt = RouteProp<InBodyStackParamList, 'RiskScreen'>;

const TEAL = '#00C2A8';

const DISEASE_KOR: Record<string, string> = {
  hypertension: '고혈압',
  diabetes: '당뇨병',
  cholesterol: '고콜레스테롤',
  triglyceride: '고중성지방',
};

const RISK_CONFIG: Record<RiskLevel, { label: string; bigLabel: string; color: string; bg: string }> = {
  low:    { label: '낮은 위험', bigLabel: '낮음', color: '#009688', bg: '#E0F2F1' },
  medium: { label: '중간 위험', bigLabel: '중간 위험', color: '#FF8F00', bg: '#FFF8E1' },
  high:   { label: '높은 위험', bigLabel: '높음', color: '#C62828', bg: '#FFEBEE' },
};

const IMPROVE_ITEMS: Record<string, { icon: keyof typeof Ionicons.glyphMap; title: string; desc: string }[]> = {
  hypertension: [
    { icon: 'nutrition-outline', title: '나트륨 섭취 줄이기', desc: '하루 나트륨 권장량 2,000mg 이하로 줄여주세요.' },
    { icon: 'fitness-outline', title: '규칙적인 유산소 운동', desc: '주 3회, 30분 이상 걷기나 가벼운 조깅을 권장합니다.' },
    { icon: 'body-outline', title: '적정 체중 유지', desc: '현재 BMI 수치로 기준으로 목표 체중 관리가 필요합니다.' },
    { icon: 'ban-outline', title: '금연 및 음주 제한', desc: '흡연과 과도한 음주는 혈압을 높일 수 있습니다.' },
  ],
  diabetes: [
    { icon: 'nutrition-outline', title: '혈당 관리 식단', desc: '정제 탄수화물 섭취를 줄이고 식이섬유를 늘리세요.' },
    { icon: 'fitness-outline', title: '규칙적인 운동', desc: '매일 30분 이상 유산소 운동으로 혈당을 조절하세요.' },
    { icon: 'body-outline', title: '체중 감량', desc: '체중을 5~10% 줄이면 혈당 개선에 효과적입니다.' },
    { icon: 'medkit-outline', title: '정기 혈당 체크', desc: '공복 혈당을 정기적으로 측정해 추세를 파악하세요.' },
  ],
  cholesterol: [
    { icon: 'nutrition-outline', title: '포화지방 줄이기', desc: '육류와 유제품의 포화지방 섭취를 제한하세요.' },
    { icon: 'fitness-outline', title: '유산소 운동', desc: '규칙적인 운동은 HDL(좋은 콜레스테롤)을 높입니다.' },
    { icon: 'leaf-outline', title: '식이섬유 섭취', desc: '채소, 과일, 통곡물로 식이섬유를 충분히 섭취하세요.' },
    { icon: 'ban-outline', title: '음주 제한', desc: '과도한 음주는 중성지방 수치를 높일 수 있습니다.' },
  ],
  triglyceride: [
    { icon: 'nutrition-outline', title: '당분 섭취 줄이기', desc: '설탕, 과당이 많은 음식과 음료를 피하세요.' },
    { icon: 'fitness-outline', title: '규칙적인 운동', desc: '유산소 운동은 중성지방 감소에 효과적입니다.' },
    { icon: 'ban-outline', title: '알코올 제한', desc: '음주는 중성지방 합성을 증가시킵니다.' },
    { icon: 'fish-outline', title: '오메가-3 섭취', desc: '등 푸른 생선이나 오메가-3 보충제 섭취를 고려하세요.' },
  ],
};

const DEFAULT_IMPROVE = [
  { icon: 'fitness-outline' as keyof typeof Ionicons.glyphMap, title: '규칙적인 운동', desc: '주 3~5회 30분 이상 유산소 운동을 권장합니다.' },
  { icon: 'nutrition-outline' as keyof typeof Ionicons.glyphMap, title: '균형 잡힌 식단', desc: '채소와 단백질 위주의 건강한 식단을 유지하세요.' },
  { icon: 'moon-outline' as keyof typeof Ionicons.glyphMap, title: '충분한 수면', desc: '하루 7~8시간의 규칙적인 수면이 중요합니다.' },
];

function DiseaseCard({ risk }: { risk: DiseaseRisk }) {
  const cfg = RISK_CONFIG[risk.risk_level];
  const items = IMPROVE_ITEMS[risk.disease] ?? DEFAULT_IMPROVE;
  const diseaseName = DISEASE_KOR[risk.disease] ?? risk.disease;
  const pct = (risk.probability * 100).toFixed(1);

  return (
    <View style={styles.diseaseBlock}>
      <Text style={styles.diseaseTitle}>{diseaseName} 위험도</Text>
      <Text style={styles.diseaseDesc}>
        이 결과는 귀하의 건강 인덱스 위치 등에 대한 위험도 예측입니다. 질병의 진단이 아닌 참고용입니다.
      </Text>

      {/* 현재 위험도 카드 */}
      <View style={styles.currentCard}>
        <Text style={styles.currentLabel}>현재 위험도</Text>
        <Text style={[styles.currentLevel, { color: cfg.color }]}>{cfg.bigLabel}</Text>
        <Text style={styles.currentPct}>위험 확률 {pct}%</Text>
        {risk.top_factors.length > 0 && (
          <Text style={styles.currentFactors}>
            주요 요인: {risk.top_factors.slice(0, 2).map((f) => f.feature_kor).join(', ')}
          </Text>
        )}
      </View>

      {/* 위험도 변화 추이 (placeholder) */}
      <Text style={styles.subTitle}>위험도 변화 추이</Text>
      <View style={styles.trendBox}>
        <Text style={styles.trendPlaceholder}>{diseaseName} 위험도 추이 그래프</Text>
      </View>
      <View style={styles.trendLabels}>
        {['3개월 전', '2개월 전', '1개월 전', '현재'].map((label, i) => (
          <View key={i} style={styles.trendLabelItem}>
            <Text style={styles.trendLabelTime}>{label}</Text>
            <Text style={[styles.trendLabelValue, i === 3 && { color: cfg.color }]}>
              {i === 3 ? cfg.bigLabel : '낮음'}
            </Text>
          </View>
        ))}
      </View>

      {/* 개선 건강사항 */}
      <Text style={styles.subTitle}>개선 건강사항</Text>
      {items.map((item) => (
        <View key={item.title} style={styles.improveCard}>
          <View style={styles.improveIcon}>
            <Ionicons name={item.icon} size={20} color={TEAL} />
          </View>
          <View style={styles.improveText}>
            <Text style={styles.improveTitle}>{item.title}</Text>
            <Text style={styles.improveDesc}>{item.desc}</Text>
          </View>
        </View>
      ))}
    </View>
  );
}

export default function RiskScreen() {
  const nav = useNavigation<Nav>();
  const { id, risks } = useRoute<Rt>().params;

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
      {risks.map((risk) => (
        <DiseaseCard key={risk.disease} risk={risk} />
      ))}

      {/* 생활습관 맞춤 가이드 */}
      <View style={styles.lifestyleCard}>
        <Text style={styles.lifestyleTitle}>나의 생활 패턴 분석</Text>
        <Text style={styles.lifestyleDesc}>
          수면 6시간 미만, 고염식 식단, 운동 부족성 위험도를 높이고 있습니다.
        </Text>
        <View style={styles.lifestyleBtns}>
          {['수면 개선 팁', '식단 조절 팁', '운동 루틴'].map((label) => (
            <TouchableOpacity key={label} style={styles.lifestyleBtn}>
              <Text style={styles.lifestyleBtnText}>{label}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* 이번 달 실천 목표 */}
      <View style={styles.goalCard}>
        <Text style={styles.goalTitle}>이번 달 실천 목표</Text>
        {['걷기 목표 설정 (분/일)', '저염식 식단', '월일 매일 기록하기'].map((goal) => (
          <View key={goal} style={styles.goalRow}>
            <Ionicons name="ellipse-outline" size={10} color="#888" />
            <Text style={styles.goalText}>{goal}</Text>
          </View>
        ))}
      </View>

      <TouchableOpacity
        style={styles.reportBtn}
        onPress={() => nav.navigate('ReportScreen', { id })}
      >
        <Text style={styles.reportBtnText}>생활습관 추천 보기</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { padding: 20, paddingBottom: 40 },

  diseaseBlock: { marginBottom: 24 },
  diseaseTitle: { fontSize: 20, fontWeight: '700', color: '#1A1A1A', marginBottom: 8 },
  diseaseDesc: { fontSize: 13, color: '#888', lineHeight: 19, marginBottom: 16 },

  currentCard: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#F0F0F0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  currentLabel: { fontSize: 12, color: '#999', marginBottom: 6 },
  currentLevel: { fontSize: 26, fontWeight: '800', marginBottom: 6 },
  currentPct: { fontSize: 14, color: '#555', marginBottom: 4 },
  currentFactors: { fontSize: 13, color: '#888' },

  subTitle: { fontSize: 16, fontWeight: '700', color: '#1A1A1A', marginBottom: 12 },

  trendBox: {
    backgroundColor: '#F5F5F5',
    borderRadius: 12,
    height: 120,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#E5E5E5',
  },
  trendPlaceholder: { fontSize: 13, color: '#AAAAAA' },
  trendLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
    paddingHorizontal: 4,
  },
  trendLabelItem: { alignItems: 'center' },
  trendLabelTime: { fontSize: 11, color: '#999', marginBottom: 3 },
  trendLabelValue: { fontSize: 12, fontWeight: '700', color: '#555' },

  improveCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 14,
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 8,
    borderWidth: 1,
    borderColor: '#F0F0F0',
  },
  improveIcon: {
    width: 38,
    height: 38,
    borderRadius: 10,
    backgroundColor: '#E0F2F1',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  improveText: { flex: 1 },
  improveTitle: { fontSize: 14, fontWeight: '700', color: '#1A1A1A', marginBottom: 3 },
  improveDesc: { fontSize: 13, color: '#666', lineHeight: 18 },

  lifestyleCard: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#F0F0F0',
  },
  lifestyleTitle: { fontSize: 15, fontWeight: '700', color: '#1A1A1A', marginBottom: 6 },
  lifestyleDesc: { fontSize: 13, color: '#666', lineHeight: 19, marginBottom: 14 },
  lifestyleBtns: { flexDirection: 'row', gap: 8 },
  lifestyleBtn: {
    backgroundColor: '#1A1A1A',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  lifestyleBtnText: { color: '#fff', fontSize: 12, fontWeight: '600' },

  goalCard: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#F0F0F0',
  },
  goalTitle: { fontSize: 15, fontWeight: '700', color: '#1A1A1A', marginBottom: 10 },
  goalRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: 8 },
  goalText: { fontSize: 13, color: '#555' },

  reportBtn: {
    backgroundColor: TEAL,
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
  },
  reportBtnText: { color: '#fff', fontSize: 16, fontWeight: '700' },
});
