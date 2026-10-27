import React, { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { api } from '../lib/api';
import type { Gender, UserUpdate } from '../types/models';

const TEAL = '#00C2A8';

// 'unknown' = 모름 (서버에는 null로 저장), null = 아직 안 고름
type Answer = boolean | 'unknown' | null;

const DISEASES = [
  { key: 'hypertension', label: '고혈압' },
  { key: 'diabetes', label: '당뇨병' },
  { key: 'dyslipidemia', label: '고지혈증 (이상지질혈증)' },
] as const;

type DiseaseKey = (typeof DISEASES)[number]['key'];
type Answers = Record<DiseaseKey, Answer>;

const EMPTY: Answers = { hypertension: null, diabetes: null, dyslipidemia: null };

const CURRENT_YEAR = new Date().getFullYear();

// 백엔드 규칙: 모름은 null로 저장
function toServer(answer: Answer): boolean | null {
  return answer === 'unknown' ? null : answer;
}

// 4자리 숫자이고 1900년 ~ 올해 사이일 때만 유효
function parseBirthYear(text: string): number | null {
  if (!/^\d{4}$/.test(text)) return null;
  const year = Number(text);
  return year >= 1900 && year <= CURRENT_YEAR ? year : null;
}

interface Props {
  onDone: () => void;
}

export default function HealthSurveyScreen({ onDone }: Props) {
  const [gender, setGender] = useState<Gender | null>(null);
  const [birthYearText, setBirthYearText] = useState('');
  const [family, setFamily] = useState<Answers>(EMPTY);
  const [busy, setBusy] = useState(false);

  const birthYear = parseBirthYear(birthYearText);
  const showBirthYearError = birthYearText.length === 4 && birthYear === null;
  const allAnswered =
    gender !== null &&
    birthYear !== null &&
    Object.values(family).every((v) => v !== null);
  const canSubmit = allAnswered && !busy;

  const onSubmit = async () => {
    if (!allAnswered) return;
    const payload: UserUpdate = {
      gender,
      birth_year: birthYear,
      fh_hypertension: toServer(family.hypertension),
      fh_diabetes: toServer(family.diabetes),
      fh_dyslipidemia: toServer(family.dyslipidemia),
    };
    setBusy(true);
    try {
      await api.updateMe(payload);
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
        <Text style={styles.headerSubtitle}>
          정확한 위험도 분석을 위해 알려 주세요
        </Text>
      </View>

      <View style={styles.form}>
        <Text style={styles.sectionTitle}>기본 정보</Text>
        <Text style={styles.sectionDesc}>나이와 성별에 따라 위험도 기준이 달라요.</Text>

        <View style={styles.row}>
          <Text style={styles.rowLabel}>성별</Text>
          <View style={styles.options}>
            <Option
              label="남성"
              checked={gender === 'male'}
              onPress={() => setGender('male')}
              disabled={busy}
            />
            <Option
              label="여성"
              checked={gender === 'female'}
              onPress={() => setGender('female')}
              disabled={busy}
            />
          </View>
        </View>

        <View style={styles.row}>
          <Text style={styles.rowLabel}>출생연도</Text>
          <TextInput
            style={[styles.yearInput, showBirthYearError && styles.yearInputError]}
            placeholder="예: 2001"
            placeholderTextColor="#BBBBBB"
            keyboardType="number-pad"
            maxLength={4}
            value={birthYearText}
            onChangeText={(t) => setBirthYearText(t.replace(/[^0-9]/g, ''))}
            editable={!busy}
          />
        </View>
        {showBirthYearError && (
          <Text style={styles.errorText}>1900년부터 {CURRENT_YEAR}년 사이로 입력하세요.</Text>
        )}

        <View style={styles.sectionGap} />

        <Section
          title="가족력"
          description="부모님이나 형제자매 중 진단받은 분이 있나요?"
          answers={family}
          onChange={setFamily}
          disabled={busy}
        />

        <Text style={styles.note}>
          입력한 정보는 질병 위험도 분석에만 쓰여요.
        </Text>

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

interface SectionProps {
  title: string;
  description: string;
  answers: Answers;
  onChange: (next: Answers) => void;
  disabled: boolean;
}

function Section({ title, description, answers, onChange, disabled }: SectionProps) {
  return (
    <View>
      <Text style={styles.sectionTitle}>{title}</Text>
      <Text style={styles.sectionDesc}>{description}</Text>

      {DISEASES.map(({ key, label }) => (
        <View key={key} style={styles.diseaseRow}>
          <Text style={styles.diseaseLabel}>{label}</Text>
          <View style={styles.options}>
            <Option
              label="있다"
              checked={answers[key] === true}
              onPress={() => onChange({ ...answers, [key]: true })}
              disabled={disabled}
            />
            <Option
              label="없다"
              checked={answers[key] === false}
              onPress={() => onChange({ ...answers, [key]: false })}
              disabled={disabled}
            />
            <Option
              label="모름"
              checked={answers[key] === 'unknown'}
              onPress={() => onChange({ ...answers, [key]: 'unknown' })}
              disabled={disabled}
            />
          </View>
        </View>
      ))}
    </View>
  );
}

interface OptionProps {
  label: string;
  checked: boolean;
  onPress: () => void;
  disabled: boolean;
}

function Option({ label, checked, onPress, disabled }: OptionProps) {
  return (
    <TouchableOpacity
      style={[styles.option, checked && styles.optionChecked]}
      onPress={onPress}
      disabled={disabled}
      accessibilityRole="checkbox"
      accessibilityState={{ checked, disabled }}
    >
      <View style={[styles.checkbox, checked && styles.checkboxChecked]}>
        {checked && <Ionicons name="checkmark" size={12} color="#fff" />}
      </View>
      <Text style={[styles.optionText, checked && styles.optionTextChecked]}>
        {label}
      </Text>
    </TouchableOpacity>
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
  headerTitle: {
    fontSize: 26,
    fontWeight: '700',
    color: '#fff',
    marginBottom: 6,
  },
  headerSubtitle: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.85)',
  },

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

  sectionTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1A1A1A',
    marginBottom: 4,
  },
  sectionDesc: {
    fontSize: 13,
    color: '#888',
    marginBottom: 12,
  },

  sectionGap: {
    height: 1,
    backgroundColor: '#F0F0F0',
    marginVertical: 20,
  },
  yearInput: {
    width: 110,
    borderWidth: 1,
    borderColor: '#E5E5E5',
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
    fontSize: 14,
    color: '#1A1A1A',
    backgroundColor: '#FAFAFA',
    textAlign: 'center',
  },
  yearInputError: {
    borderColor: '#E53935',
  },
  errorText: {
    fontSize: 12,
    color: '#E53935',
    textAlign: 'right',
    marginTop: 2,
  },

  diseaseRow: {
    paddingVertical: 10,
  },
  diseaseLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },

  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 10,
  },
  rowLabel: {
    flex: 1,
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginRight: 8,
  },
  options: {
    flexDirection: 'row',
    gap: 8,
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#E5E5E5',
    borderRadius: 10,
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: '#FAFAFA',
  },
  optionChecked: {
    borderColor: TEAL,
    backgroundColor: '#E6F9F6',
  },
  checkbox: {
    width: 18,
    height: 18,
    borderRadius: 4,
    borderWidth: 1.5,
    borderColor: '#CCCCCC',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 6,
  },
  checkboxChecked: {
    backgroundColor: TEAL,
    borderColor: TEAL,
  },
  optionText: {
    fontSize: 13,
    color: '#555',
  },
  optionTextChecked: {
    color: '#1A1A1A',
    fontWeight: '600',
  },

  note: {
    fontSize: 12,
    color: '#AAAAAA',
    marginTop: 20,
    marginBottom: 16,
  },

  primaryButton: {
    borderRadius: 12,
    paddingVertical: 15,
    alignItems: 'center',
  },
  primaryButtonActive: { backgroundColor: TEAL },
  primaryButtonInactive: { backgroundColor: '#9E9E9E' },
  primaryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
});
