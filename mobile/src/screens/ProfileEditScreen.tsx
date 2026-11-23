import React, { useEffect, useState } from 'react';
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
import { useNavigation } from '@react-navigation/native';

import HealthProfileForm, {
  EMPTY_HEALTH_PROFILE,
  fromProfile,
  isHealthProfileComplete,
  toHealthPayload,
} from '../components/HealthProfileForm';
import { api } from '../lib/api';

const TEAL = '#00C2A8';
const NICKNAME_MAX = 30; // 백엔드 제한과 같음

export default function ProfileEditScreen() {
  const navigation = useNavigation();
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [nickname, setNickname] = useState('');
  const [goal, setGoal] = useState('');
  const [health, setHealth] = useState(EMPTY_HEALTH_PROFILE);

  useEffect(() => {
    api
      .getMe()
      .then((me) => {
        setNickname(me.nickname ?? '');
        setGoal(me.fitness_goal ?? '');
        setHealth(fromProfile(me));
      })
      .catch((e: unknown) => {
        setLoadError(e instanceof Error ? e.message : '프로필을 불러오지 못했습니다.');
      })
      .finally(() => setLoading(false));
  }, []);

  const canSave = isHealthProfileComplete(health) && !busy;

  const onSave = async () => {
    if (!canSave) return;
    setBusy(true);
    try {
      await api.updateMe({
        nickname: nickname.trim() || null,
        fitness_goal: goal.trim() || null,
        ...toHealthPayload(health),
      });
      Alert.alert('저장 완료', '프로필을 저장했어요.', [
        { text: '확인', onPress: () => navigation.goBack() },
      ]);
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : '저장에 실패했습니다.';
      Alert.alert('저장 실패', message);
    } finally {
      setBusy(false);
    }
  };

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={TEAL} />
      </View>
    );
  }

  if (loadError) {
    return (
      <View style={styles.center}>
        <Text style={styles.loadError}>{loadError}</Text>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView style={styles.flex} contentContainerStyle={styles.content}>
        <View style={styles.card}>
          <Text style={styles.sectionTitle}>내 정보</Text>

          <Text style={styles.label}>이름</Text>
          <TextInput
            style={styles.input}
            placeholder="이름을 입력하세요"
            placeholderTextColor="#BBBBBB"
            maxLength={NICKNAME_MAX}
            value={nickname}
            onChangeText={setNickname}
            editable={!busy}
          />

          <Text style={styles.label}>운동 목표</Text>
          <TextInput
            style={styles.input}
            placeholder="예: 체지방 줄이기, 근육량 늘리기"
            placeholderTextColor="#BBBBBB"
            value={goal}
            onChangeText={setGoal}
            editable={!busy}
          />
          <Text style={styles.hint}>AI 리포트의 생활 가이드에 반영돼요.</Text>
        </View>

        <View style={styles.card}>
          <HealthProfileForm value={health} onChange={setHealth} disabled={busy} />
          <Text style={styles.hint}>바꾼 정보는 다음 분석부터 적용돼요.</Text>
        </View>

        <TouchableOpacity
          style={[styles.saveButton, canSave ? styles.saveActive : styles.saveInactive]}
          onPress={onSave}
          disabled={!canSave}
        >
          {busy ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.saveText}>저장</Text>
          )}
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { padding: 16, paddingBottom: 40 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#F5F5F5', padding: 24 },
  loadError: { fontSize: 14, color: '#E53935', textAlign: 'center' },

  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 8,
    elevation: 3,
  },
  sectionTitle: { fontSize: 16, fontWeight: '700', color: '#1A1A1A', marginBottom: 12 },
  label: { fontSize: 14, fontWeight: '600', color: '#333', marginBottom: 6, marginTop: 8 },
  input: {
    borderWidth: 1,
    borderColor: '#E5E5E5',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 11,
    fontSize: 14,
    color: '#1A1A1A',
    backgroundColor: '#FAFAFA',
  },
  hint: { fontSize: 12, color: '#AAAAAA', marginTop: 12 },

  saveButton: { borderRadius: 12, paddingVertical: 15, alignItems: 'center' },
  saveActive: { backgroundColor: TEAL },
  saveInactive: { backgroundColor: '#9E9E9E' },
  saveText: { color: '#fff', fontSize: 16, fontWeight: '700' },
});
