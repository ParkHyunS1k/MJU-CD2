import React, { useState } from 'react';
import { useNavigation } from '@react-navigation/native';
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
import type { InBodyCreateInput } from '../types/models';

type FieldKey = keyof Omit<InBodyCreateInput, 'measured_at' | 'image_url'>;

const FIELDS: { key: FieldKey; label: string; required?: boolean }[] = [
  { key: 'weight', label: '체중 (kg)', required: true },
  { key: 'skeletal_muscle_mass', label: '골격근량 (kg)' },
  { key: 'body_fat_mass', label: '체지방량 (kg)' },
  { key: 'body_fat_percentage', label: '체지방률 (%)' },
  { key: 'body_water', label: '체수분 (L)' },
  { key: 'protein', label: '단백질 (kg)' },
  { key: 'minerals', label: '무기질 (kg)' },
  { key: 'bmi', label: 'BMI' },
  { key: 'basal_metabolic_rate', label: '기초대사량 (kcal)' },
  { key: 'abs_fat', label: '복부지방' },
  { key: 'fat_level', label: '체지방 레벨' },
];

export default function InBodyCreateScreen() {
  const nav = useNavigation();
  const [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);

  const onSubmit = async () => {
    const weightStr = values.weight?.trim();
    if (!weightStr) {
      Alert.alert('체중을 입력해주세요.');
      return;
    }
    const payload: InBodyCreateInput = { weight: Number(weightStr) };
    if (Number.isNaN(payload.weight) || payload.weight <= 0) {
      Alert.alert('올바른 체중을 입력해주세요.');
      return;
    }
    for (const { key } of FIELDS) {
      if (key === 'weight') continue;
      const v = values[key]?.trim();
      if (!v) continue;
      const num = Number(v);
      if (Number.isNaN(num)) {
        Alert.alert(`${key} 값이 올바르지 않습니다.`);
        return;
      }
      (payload as unknown as Record<string, number>)[key] = num;
    }
    payload.measured_at = new Date().toISOString();

    setBusy(true);
    try {
      await api.createInBody(payload);
      nav.goBack();
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : '저장 실패';
      Alert.alert('오류', message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={{ flex: 1 }}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView contentContainerStyle={styles.container}>
        {FIELDS.map(({ key, label, required }) => (
          <View key={key} style={styles.field}>
            <Text style={styles.label}>
              {label}
              {required ? ' *' : ''}
            </Text>
            <TextInput
              style={styles.input}
              keyboardType="decimal-pad"
              value={values[key] ?? ''}
              onChangeText={(t) => setValues((s) => ({ ...s, [key]: t }))}
              editable={!busy}
            />
          </View>
        ))}

        <TouchableOpacity
          style={[styles.button, busy && styles.buttonDisabled]}
          onPress={onSubmit}
          disabled={busy}
        >
          {busy ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>저장</Text>
          )}
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 16, gap: 12 },
  field: { gap: 6 },
  label: { fontSize: 13, color: '#555' },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
  },
  button: {
    backgroundColor: '#1f6feb',
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 12,
  },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: '#fff', fontWeight: '600', fontSize: 16 },
});
