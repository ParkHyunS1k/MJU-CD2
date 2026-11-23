import React from 'react';
import { StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import type { Gender, UserProfile, UserUpdate } from '../types/models';

// 건강 설문과 프로필 관리 화면이 함께 쓰는 입력 폼 (성별, 출생연도, 가족력)

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

export interface HealthProfileValue {
  gender: Gender | null;
  birthYearText: string;
  family: Answers;
}

export const EMPTY_HEALTH_PROFILE: HealthProfileValue = {
  gender: null,
  birthYearText: '',
  family: { hypertension: null, diabetes: null, dyslipidemia: null },
};

const CURRENT_YEAR = new Date().getFullYear();

// 4자리 숫자이고 1900년 ~ 올해 사이일 때만 유효
function parseBirthYear(text: string): number | null {
  if (!/^\d{4}$/.test(text)) return null;
  const year = Number(text);
  return year >= 1900 && year <= CURRENT_YEAR ? year : null;
}

// 백엔드 규칙: 모름은 null로 저장
function toServer(answer: Answer): boolean | null {
  return answer === 'unknown' ? null : answer;
}

export function isHealthProfileComplete(v: HealthProfileValue): boolean {
  return (
    v.gender !== null &&
    parseBirthYear(v.birthYearText) !== null &&
    Object.values(v.family).every((a) => a !== null)
  );
}

export function toHealthPayload(v: HealthProfileValue): UserUpdate {
  return {
    gender: v.gender,
    birth_year: parseBirthYear(v.birthYearText),
    fh_hypertension: toServer(v.family.hypertension),
    fh_diabetes: toServer(v.family.diabetes),
    fh_dyslipidemia: toServer(v.family.dyslipidemia),
  };
}

// 저장된 프로필로 폼을 채운다. 설문을 마친 사용자의 가족력 null은 '모름'이다.
export function fromProfile(me: UserProfile): HealthProfileValue {
  const fromServer = (v: boolean | null): Answer => (v === null ? 'unknown' : v);
  return {
    gender: me.gender,
    birthYearText: me.birth_year ? String(me.birth_year) : '',
    family: {
      hypertension: fromServer(me.fh_hypertension),
      diabetes: fromServer(me.fh_diabetes),
      dyslipidemia: fromServer(me.fh_dyslipidemia),
    },
  };
}

interface Props {
  value: HealthProfileValue;
  onChange: (next: HealthProfileValue) => void;
  disabled: boolean;
}

export default function HealthProfileForm({ value, onChange, disabled }: Props) {
  const showBirthYearError =
    value.birthYearText.length === 4 && parseBirthYear(value.birthYearText) === null;
  const setFamily = (key: DiseaseKey, answer: Answer) =>
    onChange({ ...value, family: { ...value.family, [key]: answer } });

  return (
    <View>
      <Text style={styles.sectionTitle}>기본 정보</Text>
      <Text style={styles.sectionDesc}>나이와 성별에 따라 위험도 기준이 달라요.</Text>

      <View style={styles.row}>
        <Text style={styles.rowLabel}>성별</Text>
        <View style={styles.options}>
          <Option
            label="남성"
            checked={value.gender === 'male'}
            onPress={() => onChange({ ...value, gender: 'male' })}
            disabled={disabled}
          />
          <Option
            label="여성"
            checked={value.gender === 'female'}
            onPress={() => onChange({ ...value, gender: 'female' })}
            disabled={disabled}
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
          value={value.birthYearText}
          onChangeText={(t) => onChange({ ...value, birthYearText: t.replace(/[^0-9]/g, '') })}
          editable={!disabled}
        />
      </View>
      {showBirthYearError && (
        <Text style={styles.errorText}>1900년부터 {CURRENT_YEAR}년 사이로 입력하세요.</Text>
      )}

      <View style={styles.sectionGap} />

      <Text style={styles.sectionTitle}>가족력</Text>
      <Text style={styles.sectionDesc}>부모님이나 형제자매 중 진단받은 분이 있나요?</Text>

      {DISEASES.map(({ key, label }) => (
        <View key={key} style={styles.diseaseRow}>
          <Text style={styles.diseaseLabel}>{label}</Text>
          <View style={styles.options}>
            <Option
              label="있다"
              checked={value.family[key] === true}
              onPress={() => setFamily(key, true)}
              disabled={disabled}
            />
            <Option
              label="없다"
              checked={value.family[key] === false}
              onPress={() => setFamily(key, false)}
              disabled={disabled}
            />
            <Option
              label="모름"
              checked={value.family[key] === 'unknown'}
              onPress={() => setFamily(key, 'unknown')}
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
      <Text style={[styles.optionText, checked && styles.optionTextChecked]}>{label}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  sectionTitle: { fontSize: 16, fontWeight: '700', color: '#1A1A1A', marginBottom: 4 },
  sectionDesc: { fontSize: 13, color: '#888', marginBottom: 12 },
  sectionGap: { height: 1, backgroundColor: '#F0F0F0', marginVertical: 20 },

  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 10,
  },
  rowLabel: { flex: 1, fontSize: 14, fontWeight: '600', color: '#333', marginRight: 8 },

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
  yearInputError: { borderColor: '#E53935' },
  errorText: { fontSize: 12, color: '#E53935', textAlign: 'right', marginTop: 2 },

  diseaseRow: { paddingVertical: 10 },
  diseaseLabel: { fontSize: 14, fontWeight: '600', color: '#333', marginBottom: 8 },

  options: { flexDirection: 'row', gap: 8 },
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
  optionChecked: { borderColor: TEAL, backgroundColor: '#E6F9F6' },
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
  checkboxChecked: { backgroundColor: TEAL, borderColor: TEAL },
  optionText: { fontSize: 13, color: '#555' },
  optionTextChecked: { color: '#1A1A1A', fontWeight: '600' },
});
