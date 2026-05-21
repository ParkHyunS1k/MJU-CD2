import React, { useEffect, useState } from 'react';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

import { api } from '../lib/api';
import type { InBodyStackParamList } from '../navigation/types';
import type { AIReport, DiseaseRisk, RiskLevel } from '../types/models';

type Rt = RouteProp<InBodyStackParamList, 'ReportScreen'>;

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
    <View style={styles.riskCard}>
      <View style={styles.riskHeader}>
        <Text style={styles.riskName}>{DISEASE_KOR[risk.disease] ?? risk.disease}</Text>
        <View style={[styles.riskBadge, { backgroundColor: cfg.bg }]}>
          <Text style={[styles.riskBadgeText, { color: cfg.color }]}>{cfg.label}</Text>
        </View>
      </View>
      <View style={styles.probRow}>
        <View style={styles.probBarBg}>
          <View style={[styles.probBarFill, { width: `${risk.probability * 100}%` as any, backgroundColor: cfg.color }]} />
        </View>
        <Text style={[styles.probText, { color: cfg.color }]}>{pct}%</Text>
      </View>
    </View>
  );
}

function Section({ title, warn, children }: { title: string; warn?: boolean; children: React.ReactNode }) {
  return (
    <View style={[styles.section, warn && styles.sectionWarn]}>
      <Text style={[styles.sectionTitle, warn && styles.sectionTitleWarn]}>{title}</Text>
      {children}
    </View>
  );
}

function BulletItem({ text }: { text: string }) {
  return (
    <View style={styles.bulletRow}>
      <Text style={styles.bullet}>•</Text>
      <Text style={styles.bulletText}>{text}</Text>
    </View>
  );
}

export default function ReportScreen() {
  const { id } = useRoute<Rt>().params;
  const nav = useNavigation();
  const [report, setReport] = useState<AIReport | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getInBody(id)
      .then((data) => setReport(data.ai_report))
      .catch((e) => Alert.alert('오류', e instanceof Error ? e.message : '불러오기 실패'))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#4f6ef7" />
      </View>
    );
  }

  if (!report) {
    return (
      <View style={styles.center}>
        <Text style={styles.empty}>리포트가 없습니다.</Text>
      </View>
    );
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>AI 건강 리포트</Text>

      {report.disease_risks?.length > 0 && (
        <>
          <Text style={styles.sectionHeader}>성인병 위험도</Text>
          {report.disease_risks.map((r) => (
            <RiskCard key={r.disease} risk={r} />
          ))}
        </>
      )}

      <Section title="🏃 생활습관 추천">
        {report.lifestyle_tips.map((tip, i) => (
          <BulletItem key={i} text={tip} />
        ))}
      </Section>

      <Section title="💊 영양제 추천">
        {report.supplement_tips.map((tip, i) => (
          <BulletItem key={i} text={tip} />
        ))}
      </Section>

      {report.cautions ? (
        <Section title="⚠️ 주의사항" warn>
          <Text style={styles.cautionText}>{report.cautions}</Text>
        </Section>
      ) : null}

      <TouchableOpacity
        style={styles.homeButton}
        onPress={() => nav.navigate('InBodyList' as never)}
      >
        <Text style={styles.homeButtonText}>홈으로</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  empty: { color: '#999', fontSize: 15 },
  container: { padding: 20, gap: 16 },
  title: { fontSize: 22, fontWeight: '800', color: '#1a1a2e', marginBottom: 4 },
  sectionHeader: { fontSize: 15, fontWeight: '700', color: '#555' },
  riskCard: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 14,
    gap: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 6,
    elevation: 2,
  },
  riskHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  riskName: { fontSize: 16, fontWeight: '700', color: '#1a1a2e' },
  riskBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 20 },
  riskBadgeText: { fontSize: 13, fontWeight: '700' },
  probRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  probBarBg: { flex: 1, height: 8, backgroundColor: '#f0f0f0', borderRadius: 4, overflow: 'hidden' },
  probBarFill: { height: '100%', borderRadius: 4 },
  probText: { fontSize: 14, fontWeight: '700', width: 44, textAlign: 'right' },
  section: {
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 16,
    gap: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 6,
    elevation: 2,
  },
  sectionWarn: { backgroundColor: '#fffbea', borderWidth: 1, borderColor: '#f0c674' },
  sectionTitle: { fontSize: 16, fontWeight: '700', color: '#1a1a2e', marginBottom: 4 },
  sectionTitleWarn: { color: '#92400e' },
  bulletRow: { flexDirection: 'row', gap: 8 },
  bullet: { color: '#4f6ef7', fontSize: 15, lineHeight: 22 },
  bulletText: { flex: 1, fontSize: 14, color: '#333', lineHeight: 22 },
  cautionText: { fontSize: 14, color: '#92400e', lineHeight: 22 },
  homeButton: {
    backgroundColor: '#f0f4f8',
    padding: 14,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 4,
  },
  homeButtonText: { color: '#444', fontWeight: '600', fontSize: 15 },
});
