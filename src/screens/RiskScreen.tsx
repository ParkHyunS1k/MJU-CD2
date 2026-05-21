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

import type { InBodyStackParamList } from '../navigation/types';
import type { DiseaseRisk, RiskLevel } from '../types/models';

type Nav = NativeStackNavigationProp<InBodyStackParamList, 'RiskScreen'>;
type Rt = RouteProp<InBodyStackParamList, 'RiskScreen'>;

const DISEASE_KOR: Record<string, string> = {
  hypertension: '고혈압',
  diabetes: '당뇨',
  cholesterol: '고콜레스테롤',
  triglyceride: '고중성지방',
};

const RISK_CONFIG: Record<RiskLevel, { label: string; color: string; bg: string }> = {
  low:    { label: '낮음', color: '#1a7f37', bg: '#dcfce7' },
  medium: { label: '주의', color: '#9a6700', bg: '#fef9c3' },
  high:   { label: '높음', color: '#b91c1c', bg: '#fee2e2' },
};

function RiskCard({ risk }: { risk: DiseaseRisk }) {
  const cfg = RISK_CONFIG[risk.risk_level];
  const pct = (risk.probability * 100).toFixed(1);

  return (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <Text style={styles.diseaseName}>
          {DISEASE_KOR[risk.disease] ?? risk.disease}
        </Text>
        <View style={[styles.badge, { backgroundColor: cfg.bg }]}>
          <Text style={[styles.badgeText, { color: cfg.color }]}>{cfg.label}</Text>
        </View>
      </View>

      <View style={styles.probRow}>
        <View style={styles.probBarBg}>
          <View
            style={[
              styles.probBarFill,
              { width: `${risk.probability * 100}%` as any, backgroundColor: cfg.color },
            ]}
          />
        </View>
        <Text style={[styles.probText, { color: cfg.color }]}>{pct}%</Text>
      </View>

      {risk.top_factors.length > 0 && (
        <View style={styles.factors}>
          <Text style={styles.factorTitle}>주요 요인</Text>
          {risk.top_factors.map((f, i) => (
            <View key={i} style={styles.factorRow}>
              <Text style={styles.factorArrow}>
                {f.direction === 'up' ? '▲' : '▼'}
              </Text>
              <Text style={styles.factorName}>{f.feature_kor}</Text>
              <Text style={styles.factorValue}>{f.value}</Text>
            </View>
          ))}
        </View>
      )}
    </View>
  );
}

export default function RiskScreen() {
  const nav = useNavigation<Nav>();
  const { id, risks } = useRoute<Rt>().params;

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>성인병 위험도 분석</Text>
        <Text style={styles.subtitle}>
          인바디 수치를 기반으로 4가지 성인병 위험도를 예측했습니다
        </Text>
      </View>

      {risks.map((risk) => (
        <RiskCard key={risk.disease} risk={risk} />
      ))}

      <TouchableOpacity
        style={styles.button}
        onPress={() => nav.navigate('ReportScreen', { id })}
      >
        <Text style={styles.buttonText}>생활습관 추천 보기</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, gap: 14 },
  header: { marginBottom: 4 },
  title: { fontSize: 22, fontWeight: '800', color: '#1a1a2e', marginBottom: 6 },
  subtitle: { fontSize: 14, color: '#666', lineHeight: 20 },
  card: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 6,
    elevation: 2,
    gap: 10,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  diseaseName: { fontSize: 17, fontWeight: '700', color: '#1a1a2e' },
  badge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 20,
  },
  badgeText: { fontSize: 13, fontWeight: '700' },
  probRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  probBarBg: {
    flex: 1,
    height: 8,
    backgroundColor: '#f0f0f0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  probBarFill: { height: '100%', borderRadius: 4 },
  probText: { fontSize: 14, fontWeight: '700', width: 44, textAlign: 'right' },
  factors: { gap: 4 },
  factorTitle: { fontSize: 12, color: '#999', marginBottom: 2 },
  factorRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  factorArrow: { fontSize: 11, width: 14 },
  factorName: { fontSize: 13, color: '#444', flex: 1 },
  factorValue: { fontSize: 13, color: '#666' },
  button: {
    backgroundColor: '#4f6ef7',
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 8,
  },
  buttonText: { color: '#fff', fontWeight: '700', fontSize: 16 },
});
