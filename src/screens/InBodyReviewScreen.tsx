import React, { useCallback, useEffect, useState } from 'react';
import {
  useNavigation,
  useRoute,
  RouteProp,
} from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';

import { api } from '../lib/api';
import type { InBodyStackParamList } from '../navigation/types';
import type { InBodyRecord, InBodyUpdateInput } from '../types/models';

type Nav = NativeStackNavigationProp<InBodyStackParamList, 'InBodyReview'>;
type Rt = RouteProp<InBodyStackParamList, 'InBodyReview'>;

type FieldKey =
  | 'weight'
  | 'skeletal_muscle_mass'
  | 'body_fat_mass'
  | 'body_fat_percentage'
  | 'body_water'
  | 'protein'
  | 'minerals'
  | 'bmi'
  | 'basal_metabolic_rate'
  | 'abs_fat'
  | 'fat_level';

const FIELDS: { key: FieldKey; label: string }[] = [
  { key: 'weight', label: '체중 (kg)' },
  { key: 'skeletal_muscle_mass', label: '골격근량 (kg)' },
  { key: 'body_fat_mass', label: '체지방량 (kg)' },
  { key: 'body_fat_percentage', label: '체지방률 (%)' },
  { key: 'body_water', label: '체수분 (L)' },
  { key: 'protein', label: '단백질 (kg)' },
  { key: 'minerals', label: '무기질 (kg)' },
  { key: 'bmi', label: 'BMI' },
  { key: 'basal_metabolic_rate', label: '기초대사량 (kcal)' },
  { key: 'abs_fat', label: '복부지방률' },
  { key: 'fat_level', label: '내장지방 레벨' },
];

export default function InBodyReviewScreen() {
  const nav = useNavigation<Nav>();
  const route = useRoute<Rt>();
  const { id, confidence, notes } = route.params;

  const [record, setRecord] = useState<InBodyRecord | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);

  const load = useCallback(async () => {
    try {
      const data = await api.getInBody(id);
      setRecord(data);
      const v: Record<string, string> = {};
      for (const { key } of FIELDS) {
        const raw = data[key as keyof InBodyRecord];
        v[key] = raw === null || raw === undefined ? '' : String(raw);
      }
      setValues(v);
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : '불러오기 실패';
      Alert.alert('오류', message);
    }
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  const saveCorrections = async (): Promise<boolean> => {
    if (!record) return false;
    const patch: InBodyUpdateInput = {};
    for (const { key } of FIELDS) {
      const newStr = values[key]?.trim() ?? '';
      const orig = record[key as keyof InBodyRecord];
      const origStr = orig === null || orig === undefined ? '' : String(orig);
      if (newStr === origStr) continue;
      if (newStr === '') continue;
      const num = Number(newStr);
      if (Number.isNaN(num)) {
        Alert.alert(`${key} 값이 올바르지 않습니다.`);
        return false;
      }
      (patch as Record<string, number>)[key] = num;
    }
    if (Object.keys(patch).length === 0) return true;
    setBusy(true);
    try {
      await api.updateInBody(id, patch);
      return true;
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : '저장 실패';
      Alert.alert('오류', message);
      return false;
    } finally {
      setBusy(false);
    }
  };

  const onSaveOnly = async () => {
    const ok = await saveCorrections();
    if (ok) nav.replace('InBodyDetail', { id });
  };

  const onSaveAndAnalyze = async () => {
    const ok = await saveCorrections();
    if (!ok) return;
    setAnalyzing(true);
    try {
      const result = await api.analyzeInBody(id);
      nav.replace('RiskScreen', { id, risks: result.disease_risks ?? [] });
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : '분석 실패';
      Alert.alert('오류', message);
    } finally {
      setAnalyzing(false);
    }
  };

  if (!record) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  const lowConf = confidence < 0.7;

  return (
    <KeyboardAvoidingView
      style={{ flex: 1 }}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView contentContainerStyle={styles.container}>
        <View style={[styles.confBox, lowConf && styles.confBoxLow]}>
          <Text style={styles.confTitle}>
            추출 신뢰도: {(confidence * 100).toFixed(0)}%
          </Text>
          <Text style={styles.confBody}>
            {lowConf
              ? '일부 값이 정확하지 않을 수 있어요. 확인 후 수정해주세요.'
              : '값이 정확하게 추출됐습니다. 필요하면 수정 가능해요.'}
          </Text>
          {notes ? <Text style={styles.confNotes}>메모: {notes}</Text> : null}
        </View>

        {FIELDS.map(({ key, label }) => (
          <View key={key} style={styles.field}>
            <Text style={styles.label}>{label}</Text>
            <TextInput
              style={styles.input}
              keyboardType="decimal-pad"
              value={values[key] ?? ''}
              onChangeText={(t) => setValues((s) => ({ ...s, [key]: t }))}
              editable={!busy && !analyzing}
            />
          </View>
        ))}

        {analyzing ? (
          <View style={styles.busyBox}>
            <ActivityIndicator size="large" />
            <Text style={styles.busyText}>AI 분석 중...</Text>
          </View>
        ) : (
          <>
            <TouchableOpacity
              style={styles.primaryButton}
              onPress={onSaveAndAnalyze}
              disabled={busy}
            >
              <Text style={styles.primaryButtonText}>
                {busy ? '저장 중...' : '저장하고 AI 분석'}
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.secondaryButton}
              onPress={onSaveOnly}
              disabled={busy}
            >
              <Text style={styles.secondaryButtonText}>저장만 하기</Text>
            </TouchableOpacity>
          </>
        )}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 16, gap: 12 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  confBox: {
    padding: 12,
    backgroundColor: '#eaf5ff',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#bcdcff',
    gap: 4,
  },
  confBoxLow: {
    backgroundColor: '#fffbea',
    borderColor: '#f0c674',
  },
  confTitle: { fontSize: 14, fontWeight: '700' },
  confBody: { fontSize: 13, color: '#444' },
  confNotes: { fontSize: 12, color: '#666', marginTop: 4 },
  field: { gap: 6 },
  label: { fontSize: 13, color: '#555' },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
  },
  primaryButton: {
    backgroundColor: '#1f6feb',
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 12,
  },
  primaryButtonText: { color: '#fff', fontWeight: '600', fontSize: 16 },
  secondaryButton: {
    padding: 12,
    alignItems: 'center',
  },
  secondaryButtonText: { color: '#1f6feb', fontWeight: '600' },
  busyBox: { alignItems: 'center', gap: 12, marginTop: 24 },
  busyText: { color: '#444' },
});
