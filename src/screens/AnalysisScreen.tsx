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
import type { AIReport } from '../types/models';

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
      <ListSection title="생활습관 추천" items={report.lifestyle_tips} />
      <ListSection title="영양제 추천" items={report.supplement_tips} />
      {report.cautions ? (
        <Section title="주의사항" body={report.cautions} variant="warn" />
      ) : null}
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
  listItem: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 4,
  },
  bullet: { fontSize: 14, color: '#1f6feb', lineHeight: 21 },
  listText: { fontSize: 14, lineHeight: 21, color: '#24292e', flex: 1 },
});
