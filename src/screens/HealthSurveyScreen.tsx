import React, { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

import HealthProfileForm, {
  EMPTY_HEALTH_PROFILE,
  isHealthProfileComplete,
  toHealthPayload,
} from '../components/HealthProfileForm';
import { api } from '../lib/api';

const TEAL = '#00C2A8';

interface Props {
  onDone: () => void;
}

export default function HealthSurveyScreen({ onDone }: Props) {
  const [form, setForm] = useState(EMPTY_HEALTH_PROFILE);
  const [busy, setBusy] = useState(false);

  const complete = isHealthProfileComplete(form);
  const canSubmit = complete && !busy;

  const onSubmit = async () => {
    if (!complete) return;
    setBusy(true);
    try {
      await api.updateMe(toHealthPayload(form));
      onDone();
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : '저장에 실패했습니다.';
      Alert.alert('저장 실패', message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <ScrollView style={styles.flex} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>건강 설문</Text>
        <Text style={styles.headerSubtitle}>정확한 위험도 분석을 위해 알려 주세요</Text>
      </View>

      <View style={styles.form}>
        <HealthProfileForm value={form} onChange={setForm} disabled={busy} />

        <Text style={styles.note}>입력한 정보는 질병 위험도 분석에만 쓰여요.</Text>

        <TouchableOpacity
          style={[
            styles.primaryButton,
            canSubmit ? styles.primaryButtonActive : styles.primaryButtonInactive,
          ]}
          onPress={onSubmit}
          disabled={!canSubmit}
        >
          {busy ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.primaryButtonText}>저장하고 시작하기</Text>
          )}
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { flexGrow: 1 },

  header: {
    backgroundColor: TEAL,
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
    paddingTop: 70,
    paddingBottom: 32,
    alignItems: 'center',
  },
  headerTitle: { fontSize: 26, fontWeight: '700', color: '#fff', marginBottom: 6 },
  headerSubtitle: { fontSize: 14, color: 'rgba(255,255,255,0.85)' },

  form: {
    backgroundColor: '#fff',
    marginHorizontal: 16,
    marginTop: 20,
    borderRadius: 16,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 3,
    marginBottom: 32,
  },

  note: { fontSize: 12, color: '#AAAAAA', marginTop: 20, marginBottom: 16 },

  primaryButton: { borderRadius: 12, paddingVertical: 15, alignItems: 'center' },
  primaryButtonActive: { backgroundColor: TEAL },
  primaryButtonInactive: { backgroundColor: '#9E9E9E' },
  primaryButtonText: { color: '#fff', fontSize: 16, fontWeight: '700' },
});
