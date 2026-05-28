import React, { useState } from 'react';
import {
  ActivityIndicator,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

import { api } from '../lib/api';
import type {
  AIReport,
  BodyWatchPoint,
  LifestylePlanItem,
  RiskLevel,
  SupplementCandidate,
} from '../types/models';

const PRIORITY_LABEL: Record<RiskLevel, string> = {
  low: '낮음',
  medium: '주의',
  high: '높음',
};

const CATEGORY_LABEL: Record<LifestylePlanItem['category'], string> = {
  nutrition: '식사',
  exercise: '운동',
  sleep: '수면',
  habit: '습관',
};

const EVIDENCE_LABEL: Record<SupplementCandidate['evidence_level'], string> = {
  general: '일반',
  moderate: '보통',
  limited: '제한적',
};

export default function AnalysisScreen() {
  const [report, setReport] = useState<AIReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await api.analyzeInBody();
      setReport(res);
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : '분석 실패';
      setError(message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>AI 건강 분석</Text>
      <Text style={styles.subtitle}>
        최근 두 번의 InBody 기록을 비교해 Gemini가 분석 리포트를 생성합니다.
      </Text>

      <TouchableOpacity
        style={[styles.button, busy && styles.buttonDisabled]}
        onPress={run}
        disabled={busy}
      >
        {busy ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>
            {report ? '다시 분석하기' : '분석 시작'}
          </Text>
        )}
      </TouchableOpacity>

      {error ? (
        <View style={styles.errorBox}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      ) : null}

      {report ? <ReportView report={report} /> : null}
    </ScrollView>
  );
}

function ReportView({ report }: { report: AIReport }) {
  return (
    <View style={{ gap: 12 }}>
      <Section title={report.summary.title} body={report.summary.overall_message} />
      <ListSection title="핵심 관리 포인트" items={report.summary.key_focus} />

      <View style={styles.card}>
        <Text style={styles.cardTitle}>인바디 해석</Text>
        {report.body_composition_report.strengths.map((item, idx) => (
          <MetricInsightView key={`strength-${idx}`} metric={item.metric} message={item.message} />
        ))}
        {report.body_composition_report.watch_points.map((item, idx) => (
          <WatchPointView key={`watch-${idx}`} item={item} />
        ))}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>생활습관 계획</Text>
        {report.lifestyle_plan.map((item, idx) => (
          <PlanItemView key={idx} item={item} />
        ))}
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>보충제 후보</Text>
        {report.supplement_candidates.map((item, idx) => (
          <SupplementView key={idx} item={item} />
        ))}
      </View>

      <View style={[styles.card, styles.cardWarn]}>
        <Text style={[styles.cardTitle, styles.cardTitleWarn]}>주의사항</Text>
        <Text style={[styles.cardBody, styles.cardBodyWarn]}>
          {report.cautions.medical_disclaimer}
        </Text>
        {report.cautions.consult_doctor_if.map((item, idx) => (
          <View key={idx} style={styles.listItem}>
            <Text style={[styles.bullet, styles.bulletWarn]}>•</Text>
            <Text style={[styles.listText, styles.cardBodyWarn]}>{item}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

function Section({
  title,
  body,
  variant,
}: {
  title: string;
  body: string;
  variant?: 'warn';
}) {
  const isWarn = variant === 'warn';
  return (
    <View style={[styles.card, isWarn && styles.cardWarn]}>
      <Text style={[styles.cardTitle, isWarn && styles.cardTitleWarn]}>
        {title}
      </Text>
      <Text style={[styles.cardBody, isWarn && styles.cardBodyWarn]}>
        {body}
      </Text>
    </View>
  );
}

function ListSection({ title, items }: { title: string; items: string[] }) {
  return (
    <View style={styles.card}>
      <Text style={styles.cardTitle}>{title}</Text>
      {items.map((item, idx) => (
        <View key={idx} style={styles.listItem}>
          <Text style={styles.bullet}>•</Text>
          <Text style={styles.listText}>{item}</Text>
        </View>
      ))}
    </View>
  );
}

function MetricInsightView({ metric, message }: { metric: string; message: string }) {
  return (
    <View style={styles.subBlock}>
      <Text style={styles.subTitle}>{metric}</Text>
      <Text style={styles.cardBody}>{message}</Text>
    </View>
  );
}

function WatchPointView({ item }: { item: BodyWatchPoint }) {
  return (
    <View style={styles.subBlock}>
      <View style={styles.inlineHeader}>
        <Text style={styles.subTitle}>{item.metric}</Text>
        <Text style={styles.metaText}>{PRIORITY_LABEL[item.priority]}</Text>
      </View>
      <Text style={styles.cardBody}>{item.current_value}</Text>
      <Text style={styles.cardBody}>{item.interpretation}</Text>
    </View>
  );
}

function PlanItemView({ item }: { item: LifestylePlanItem }) {
  return (
    <View style={styles.subBlock}>
      <Text style={styles.subTitle}>
        {CATEGORY_LABEL[item.category]} · {item.title}
      </Text>
      <Text style={styles.cardBody}>{item.reason}</Text>
      {item.actions.map((action, idx) => (
        <View key={idx} style={styles.listItem}>
          <Text style={styles.bullet}>•</Text>
          <Text style={styles.listText}>{action}</Text>
        </View>
      ))}
    </View>
  );
}

function SupplementView({ item }: { item: SupplementCandidate }) {
  return (
    <View style={styles.subBlock}>
      <View style={styles.inlineHeader}>
        <Text style={styles.subTitle}>{item.name}</Text>
        <Text style={styles.metaText}>근거 {EVIDENCE_LABEL[item.evidence_level]}</Text>
      </View>
      <Text style={styles.cardBody}>{item.purpose}</Text>
      <Text style={styles.cardBody}>{item.why_consider}</Text>
      <Text style={styles.metaText}>관련 지표: {item.related_metrics.join(', ')}</Text>
      {item.cautions.map((caution, idx) => (
        <View key={`caution-${idx}`} style={styles.listItem}>
          <Text style={styles.bullet}>•</Text>
          <Text style={styles.listText}>{caution}</Text>
        </View>
      ))}
      {item.not_recommended_if.length > 0 ? (
        <Text style={styles.metaText}>
          피해야 할 경우: {item.not_recommended_if.join(', ')}
        </Text>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { padding: 16, gap: 16 },
  title: { fontSize: 24, fontWeight: '700' },
  subtitle: { fontSize: 14, color: '#666' },
  button: {
    backgroundColor: '#1f6feb',
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: '#fff', fontWeight: '600', fontSize: 16 },
  errorBox: {
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#ffeef0',
    borderWidth: 1,
    borderColor: '#fdb8c0',
  },
  errorText: { color: '#86181d' },
  card: {
    padding: 14,
    backgroundColor: '#f6f8fa',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e1e4e8',
    gap: 6,
  },
  cardWarn: {
    backgroundColor: '#fffbea',
    borderColor: '#f0c674',
  },
  cardTitle: { fontSize: 15, fontWeight: '700', color: '#24292e' },
  cardTitleWarn: { color: '#735c0f' },
  cardBody: { fontSize: 14, lineHeight: 21, color: '#24292e' },
  cardBodyWarn: { color: '#735c0f' },
  subBlock: {
    borderTopWidth: 1,
    borderTopColor: '#eaeef2',
    paddingTop: 8,
    gap: 4,
  },
  inlineHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 8,
    alignItems: 'center',
  },
  subTitle: { fontSize: 14, fontWeight: '700', color: '#24292e', flex: 1 },
  metaText: { fontSize: 12, lineHeight: 18, color: '#57606a' },
  listItem: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 4,
  },
  bullet: { fontSize: 14, color: '#1f6feb', lineHeight: 21 },
  bulletWarn: { color: '#735c0f' },
  listText: { fontSize: 14, lineHeight: 21, color: '#24292e', flex: 1 },
});
