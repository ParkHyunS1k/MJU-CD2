import React, { useState } from 'react';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import {
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as DocumentPicker from 'expo-document-picker';

import { api, UploadFile } from '../lib/api';
import type { InBodyStackParamList } from '../navigation/types';

type Nav = NativeStackNavigationProp<InBodyStackParamList, 'InBodyUpload'>;

const TEAL = '#00C2A8';

function guessMimeFromUri(uri: string): string {
  const lower = uri.toLowerCase();
  if (lower.endsWith('.png')) return 'image/png';
  if (lower.endsWith('.heic')) return 'image/heic';
  if (lower.endsWith('.webp')) return 'image/webp';
  if (lower.endsWith('.pdf')) return 'application/pdf';
  return 'image/jpeg';
}

export default function InBodyUploadScreen() {
  const nav = useNavigation<Nav>();
  const [busy, setBusy] = useState(false);
  const [selectedFile, setSelectedFile] = useState<UploadFile | null>(null);

  const pickFile = async () => {
    const res = await DocumentPicker.getDocumentAsync({
      type: ['image/*', 'application/pdf'],
      copyToCacheDirectory: true,
    });
    if (res.canceled) return;
    const asset = res.assets[0];
    setSelectedFile({
      uri: asset.uri,
      name: asset.name,
      type: asset.mimeType ?? guessMimeFromUri(asset.name),
    });
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      await pickFile();
      return;
    }
    setBusy(true);
    try {
      const res = await api.uploadInBody(selectedFile);
      nav.replace('InBodyReview', {
        id: res.record.id,
        confidence: res.extraction_confidence,
        notes: res.extraction_notes,
      });
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : '업로드 실패';
      Alert.alert('오류', message);
    } finally {
      setBusy(false);
    }
  };

  const INFO_ITEMS = [
    '인바디 앱 PDF 출력물',
    '인바디 측정지 사진 (JPG, PNG, HEIC)',
    '최대 파일 크기: 10MB',
    '여러 파일 동시 업로드 가능',
  ];

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>인바디 데이터 업로드</Text>
      </View>

      <View style={styles.body}>
        <Text style={styles.bodyTitle}>파일을 업로드하세요</Text>
        <Text style={styles.bodyDesc}>
          인바디 앱에서 내보낸 측정 결과 또는 인바디 측정지 사진을 업로드해 주세요.
        </Text>

        <TouchableOpacity style={styles.uploadZone} onPress={pickFile} activeOpacity={0.7}>
          <Ionicons name="cloud-upload-outline" size={40} color="#CCCCCC" />
          <Text style={styles.uploadZoneTitle}>
            {selectedFile ? selectedFile.name : '파일을 드래그하거나 클릭하여 선택'}
          </Text>
          <Text style={styles.uploadZoneSub}>PDF, JPG, PNG, HEIC 파일 지원</Text>
          {selectedFile && (
            <View style={styles.fileSelectedBadge}>
              <Ionicons name="checkmark-circle" size={14} color={TEAL} />
              <Text style={styles.fileSelectedText}>파일 선택됨</Text>
            </View>
          )}
        </TouchableOpacity>

        <View style={styles.infoBox}>
          <Text style={styles.infoTitle}>업로드 가능한 파일</Text>
          {INFO_ITEMS.map((item) => (
            <View key={item} style={styles.infoRow}>
              <Text style={styles.infoBullet}>·</Text>
              <Text style={styles.infoText}>{item}</Text>
            </View>
          ))}
        </View>
      </View>

      <View style={styles.footer}>
        <TouchableOpacity
          style={[styles.primaryBtn, selectedFile && !busy && styles.primaryBtnActive]}
          onPress={handleAnalyze}
          disabled={busy}
        >
          <Text style={styles.primaryBtnText}>
            {busy ? '업로드 중...' : '분석 시작하기'}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.secondaryBtn}
          onPress={() => nav.goBack()}
          disabled={busy}
        >
          <Text style={styles.secondaryBtnText}>나중에 하기</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { flexGrow: 1 },

  header: {
    backgroundColor: TEAL,
    paddingTop: 60,
    paddingBottom: 24,
    paddingHorizontal: 20,
    alignItems: 'center',
  },
  headerTitle: { color: '#fff', fontSize: 18, fontWeight: '700' },

  body: { flex: 1, padding: 20 },
  bodyTitle: { fontSize: 20, fontWeight: '700', color: '#1A1A1A', marginBottom: 8 },
  bodyDesc: { fontSize: 13, color: '#757575', lineHeight: 20, marginBottom: 20 },

  uploadZone: {
    backgroundColor: '#fff',
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: '#E5E5E5',
    paddingVertical: 36,
    alignItems: 'center',
    gap: 8,
    marginBottom: 16,
  },
  uploadZoneTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#555',
    textAlign: 'center',
    paddingHorizontal: 16,
  },
  uploadZoneSub: { fontSize: 12, color: '#AAAAAA' },
  fileSelectedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: 4,
  },
  fileSelectedText: { fontSize: 12, color: TEAL, fontWeight: '600' },

  infoBox: {
    backgroundColor: '#F8F8F8',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: '#EEEEEE',
  },
  infoTitle: { fontSize: 13, fontWeight: '700', color: '#333', marginBottom: 10 },
  infoRow: { flexDirection: 'row', gap: 6, marginBottom: 5 },
  infoBullet: { fontSize: 13, color: '#999' },
  infoText: { fontSize: 13, color: '#666', flex: 1 },

  footer: { padding: 20, gap: 10 },
  primaryBtn: {
    backgroundColor: '#9E9E9E',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
  },
  primaryBtnActive: { backgroundColor: TEAL },
  primaryBtnText: { color: '#fff', fontSize: 16, fontWeight: '700' },
  secondaryBtn: {
    borderRadius: 12,
    paddingVertical: 15,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#E5E5E5',
    backgroundColor: '#fff',
  },
  secondaryBtnText: { fontSize: 15, color: '#555' },
});
