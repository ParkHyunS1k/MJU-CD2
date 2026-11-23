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
  | 'fat_level'
  | 'height'
  | 'muscle_control';

type Field = { key: FieldKey; label: string; unit: string; hint?: string };

// 한 화면에 들어오도록 성격별로 묶어서 한 줄씩 보여 준다
const GROUPS: { title: string; note?: string; fields: Field[] }[] = [
  {
    title: '맞춤 분석 정보',
    note: '결과지에서 읽지 못했다면 직접 입력해 주세요. 리포트가 더 정확해져요.',
    fields: [
      { key: 'height', label: '신장', unit: 'cm' },
      { key: 'muscle_control', label: '근육조절', unit: 'kg', hint: '결과지 체중조절 칸' },
    ],
  },
  {
    title: '체성분',
    fields: [
      { key: 'weight', label: '체중', unit: 'kg' },
      { key: 'skeletal_muscle_mass', label: '골격근량', unit: 'kg' },
      { key: 'body_fat_mass', label: '체지방량', unit: 'kg' },
      { key: 'body_fat_percentage', label: '체지방률', unit: '%' },
    ],
  },
  {
    title: '비만·대사 지표',
    fields: [
      { key: 'bmi', label: 'BMI', unit: 'kg/㎡' },
      { key: 'abs_fat', label: '복부지방률', unit: '' },
      { key: 'fat_level', label: '내장지방 레벨', unit: '' },
      { key: 'basal_metabolic_rate', label: '기초대사량', unit: 'kcal' },
    ],
  },
  {
    title: '구성 성분',
    fields: [
      { key: 'body_water', label: '체수분', unit: 'L' },
      { key: 'protein', label: '단백질', unit: 'kg' },
      { key: 'minerals', label: '무기질', unit: 'kg' },
    ],
  },
];

const FIELDS: Field[] = GROUPS.flatMap((g) => g.fields);
const TEAL = '#00C2A8';

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
        const label = FIELDS.find((f) => f.key === key)?.label ?? key;
        Alert.alert(`${label} 값이 올바르지 않습니다.`);
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
      nav.replace('ReportScreen', { id });
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
  const locked = busy || analyzing;

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView
        style={styles.flex}
        contentContainerStyle={styles.container}
        // iOS에서 상단 제목 줄 밑으로 내용이 가려지지 않게 한다
        contentInsetAdjustmentBehavior="automatic"
        keyboardShouldPersistTaps="handled"
      >
        <View style={[styles.confBox, lowConf && styles.confBoxLow]}>
          <Text style={styles.confTitle}>
            추출 신뢰도 {(confidence * 100).toFixed(0)}%
            <Text style={styles.confBody}>
              {lowConf
                ? '  일부 값이 틀릴 수 있어요. 확인해 주세요.'
                : '  틀린 값이 있으면 눌러서 고치세요.'}
            </Text>
          </Text>
          {notes ? <Text style={styles.confNotes}>메모: {notes}</Text> : null}
        </View>

        {GROUPS.map((group) => (
          <View key={group.title} style={styles.card}>
            <Text style={styles.cardTitle}>{group.title}</Text>
            {group.note ? <Text style={styles.cardNote}>{group.note}</Text> : null}
            {group.fields.map(({ key, label, unit, hint }, idx) => (
              <View
                key={key}
                style={[styles.row, idx < group.fields.length - 1 && styles.rowBorder]}
              >
                <View>
                  <Text style={styles.label}>{label}</Text>
                  {hint ? <Text style={styles.hint}>{hint}</Text> : null}
                </View>
                <View style={styles.valueBox}>
                  <TextInput
                    style={styles.input}
                    keyboardType="decimal-pad"
                    value={values[key] ?? ''}
                    placeholder="-"
                    placeholderTextColor="#BBBBBB"
                    onChangeText={(t) => setValues((s) => ({ ...s, [key]: t }))}
                    editable={!locked}
                    selectTextOnFocus
                  />
                  <Text style={styles.unit}>{unit}</Text>
                </View>
              </View>
            ))}
          </View>
        ))}
      </ScrollView>

      {/* 버튼은 화면 아래에 고정해서 스크롤하지 않아도 누를 수 있게 한다 */}
      <View style={styles.bottomBar}>
        <TouchableOpacity
          style={[styles.primaryButton, locked && styles.primaryButtonBusy]}
          onPress={onSaveAndAnalyze}
          disabled={locked}
        >
          {analyzing ? (
            <View style={styles.busyRow}>
              <ActivityIndicator color="#fff" />
              <Text style={styles.primaryButtonText}>AI 분석 중… 최대 1분 걸려요</Text>
            </View>
          ) : (
            <Text style={styles.primaryButtonText}>
              {busy ? '저장 중…' : '저장하고 AI 분석'}
            </Text>
          )}
        </TouchableOpacity>
        {!analyzing && (
          <TouchableOpacity style={styles.secondaryButton} onPress={onSaveOnly} disabled={locked}>
            <Text style={styles.secondaryButtonText}>저장만 하기</Text>
          </TouchableOpacity>
        )}
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: '#F5F5F5' },
  container: { padding: 16, gap: 12 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },

  confBox: {
    paddingVertical: 10,
    paddingHorizontal: 12,
    backgroundColor: '#E6F9F6',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#B5EDE3',
  },
  confBoxLow: { backgroundColor: '#FFFBEA', borderColor: '#F0C674' },
  confTitle: { fontSize: 14, fontWeight: '700', color: '#1A1A1A' },
  confBody: { fontSize: 13, fontWeight: '400', color: '#555' },
  confNotes: { fontSize: 12, color: '#666', marginTop: 4 },

  card: {
    backgroundColor: '#fff',
    borderRadius: 14,
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  cardTitle: { fontSize: 13, fontWeight: '700', color: '#00A38D', marginBottom: 2 },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 6,
  },
  rowBorder: { borderBottomWidth: 1, borderBottomColor: '#F0F0F0' },
  label: { fontSize: 15, color: '#333' },
  hint: { fontSize: 11, color: '#AAAAAA', marginTop: 1 },
  cardNote: { fontSize: 12, color: '#888', marginBottom: 4 },
  valueBox: { flexDirection: 'row', alignItems: 'center' },
  input: {
    minWidth: 84,
    borderWidth: 1,
    borderColor: '#E5E5E5',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
    fontSize: 16,
    fontWeight: '600',
    color: '#1A1A1A',
    backgroundColor: '#FAFAFA',
    textAlign: 'right',
  },
  unit: { width: 44, marginLeft: 6, fontSize: 13, color: '#888' },

  bottomBar: {
    backgroundColor: '#fff',
    paddingHorizontal: 16,
    paddingTop: 10,
    paddingBottom: Platform.OS === 'ios' ? 28 : 12,
    borderTopWidth: 1,
    borderTopColor: '#EEEEEE',
  },
  primaryButton: {
    backgroundColor: TEAL,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  primaryButtonBusy: { opacity: 0.85 },
  primaryButtonText: { color: '#fff', fontWeight: '700', fontSize: 16 },
  busyRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  secondaryButton: { paddingVertical: 10, alignItems: 'center' },
  secondaryButtonText: { color: '#00A38D', fontWeight: '600', fontSize: 14 },
});
