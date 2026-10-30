import React, { useCallback, useState } from 'react';
import {
  useFocusEffect,
  useNavigation,
  useRoute,
  RouteProp,
} from '@react-navigation/native';
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
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { InBodyRecord } from '../types/models';

type Nav = NativeStackNavigationProp<InBodyStackParamList, 'InBodyDetail'>;
type R = RouteProp<InBodyStackParamList, 'InBodyDetail'>;

const FIELD_LABELS: { key: keyof InBodyRecord; label: string; unit?: string }[] = [
  { key: 'height', label: '신장', unit: 'cm' },
  { key: 'weight', label: '체중', unit: 'kg' },
  { key: 'skeletal_muscle_mass', label: '골격근량', unit: 'kg' },
  { key: 'body_fat_mass', label: '체지방량', unit: 'kg' },
  { key: 'body_fat_percentage', label: '체지방률', unit: '%' },
  { key: 'body_water', label: '체수분', unit: 'L' },
  { key: 'protein', label: '단백질', unit: 'kg' },
  { key: 'minerals', label: '무기질', unit: 'kg' },
  { key: 'bmi', label: 'BMI', unit: 'kg/㎡' },
  { key: 'basal_metabolic_rate', label: '기초대사량', unit: 'kcal' },
  { key: 'abs_fat', label: '복부지방률' },
  { key: 'fat_level', label: '내장지방 레벨' },
  { key: 'muscle_control', label: '근육조절', unit: 'kg' },
];

export default function InBodyDetailScreen() {
  const route = useRoute<R>();
  const nav = useNavigation<Nav>();
  const { id } = route.params;
  const [record, setRecord] = useState<InBodyRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState(false);

  const load = useCallback(async () => {
    try {
      const data = await api.getInBody(id);
      setRecord(data);
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : '불러오기 실패';
      Alert.alert('오류', message);
    } finally {
      setLoading(false);
    }
  }, [id]);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load]),
  );

  const onDelete = () => {
    Alert.alert('삭제', '이 기록을 삭제하시겠습니까?', [
      { text: '취소', style: 'cancel' },
      {
        text: '삭제',
        style: 'destructive',
        onPress: async () => {
          setDeleting(true);
          try {
            await api.deleteInBody(id);
            nav.goBack();
          } catch (e: unknown) {
            const message = e instanceof Error ? e.message : '삭제 실패';
            Alert.alert('오류', message);
            setDeleting(false);
          }
        },
      },
    ]);
  };

  if (loading || !record) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.date}>
        측정일{' '}
        {record.measured_at
          ? new Date(record.measured_at).toLocaleString()
          : '-'}
      </Text>

      {FIELD_LABELS.map(({ key, label, unit }) => {
        const value = record[key];
        if (value === null || value === undefined) return null;
        return (
          <View key={String(key)} style={styles.row}>
            <Text style={styles.rowLabel}>{label}</Text>
            <Text style={styles.rowValue}>
              {String(value)}
              {unit ? ` ${unit}` : ''}
            </Text>
          </View>
        );
      })}

      {record.ai_report ? (
        <TouchableOpacity
          style={styles.reportButton}
          onPress={() => nav.navigate('ReportScreen', { id })}
        >
          <Text style={styles.reportButtonText}>AI 리포트 보기</Text>
        </TouchableOpacity>
      ) : null}

      <TouchableOpacity
        style={[styles.deleteButton, deleting && { opacity: 0.6 }]}
        onPress={onDelete}
        disabled={deleting}
      >
        <Text style={styles.deleteText}>
          {deleting ? '삭제 중...' : '삭제'}
        </Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 16, gap: 8 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  date: { fontSize: 14, color: '#666', marginBottom: 12 },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  rowLabel: { fontSize: 15, color: '#444' },
  rowValue: { fontSize: 15, fontWeight: '600' },
  reportButton: {
    marginTop: 16,
    backgroundColor: '#00C2A8',
    padding: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  reportButtonText: { color: '#fff', fontWeight: '700', fontSize: 15 },
  deleteButton: {
    marginTop: 24,
    padding: 14,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#d73a49',
    alignItems: 'center',
  },
  deleteText: { color: '#d73a49', fontWeight: '600' },
});
