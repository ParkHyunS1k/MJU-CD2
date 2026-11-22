import React, { useEffect, useState } from 'react';
import { RouteProp, useNavigation, useRoute } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { ActivityIndicator, Alert, ScrollView, StyleSheet, Text, View } from 'react-native';

import ReportView from '../components/report/ReportView';
import { api } from '../lib/api';
import type { InBodyStackParamList } from '../navigation/types';
import type { InBodyRecord } from '../types/models';

type Rt = RouteProp<InBodyStackParamList, 'ReportScreen'>;

const TEAL = '#00C2A8';

export function formatDate(dateStr: string | null): string {
  if (!dateStr) return '-';
  const d = new Date(dateStr);
  return `${d.getFullYear()}.${d.getMonth() + 1}.${d.getDate()}`;
}

// 기록 상세 → "AI 리포트 보기", 또는 분석 직후에 열리는 리포트 화면
export default function ReportScreen() {
  const { id } = useRoute<Rt>().params;
  const nav = useNavigation<NativeStackNavigationProp<InBodyStackParamList, 'ReportScreen'>>();
  const [record, setRecord] = useState<InBodyRecord | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .getInBody(id)
      .then(setRecord)
      .catch((e) => Alert.alert('오류', e instanceof Error ? e.message : '불러오기 실패'))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color={TEAL} />
      </View>
    );
  }

  if (!record?.ai_report) {
    return (
      <View style={styles.center}>
        <Text style={styles.emptyText}>이 기록은 아직 분석하지 않았어요.</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.scroll}
      contentContainerStyle={styles.content}
      // iOS에서 상단 제목 줄 밑으로 내용이 가려지지 않게 한다
      contentInsetAdjustmentBehavior="automatic"
    >
      <ReportView
        report={record.ai_report}
        dateLabel={formatDate(record.measured_at)}
        onPressDisease={(disease) => nav.navigate('RiskScreen', { id, disease })}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { padding: 16, paddingBottom: 40 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#F5F5F5' },
  emptyText: { color: '#999', fontSize: 15 },
});
