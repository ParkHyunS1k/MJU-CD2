import React, { useState } from 'react';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import {
  ActivityIndicator,
  Alert,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as DocumentPicker from 'expo-document-picker';

import { api, UploadFile } from '../lib/api';
import type { InBodyStackParamList } from '../navigation/types';

type Nav = NativeStackNavigationProp<InBodyStackParamList, 'InBodyUpload'>;

function guessMimeFromUri(uri: string): string {
  const lower = uri.toLowerCase();
  if (lower.endsWith('.png')) return 'image/png';
  if (lower.endsWith('.heic')) return 'image/heic';
  if (lower.endsWith('.webp')) return 'image/webp';
  if (lower.endsWith('.pdf')) return 'application/pdf';
  return 'image/jpeg';
}

function fileNameFromUri(uri: string): string {
  const segs = uri.split('/');
  return segs[segs.length - 1] || 'upload.jpg';
}

const OPTIONS = [
  {
    id: 'camera',
    icon: '📷',
    label: '카메라로 촬영',
    desc: '인바디 결과지를 직접 찍어주세요',
  },
  {
    id: 'gallery',
    icon: '🖼️',
    label: '갤러리에서 선택',
    desc: '저장된 사진을 불러옵니다',
  },
  {
    id: 'pdf',
    icon: '📄',
    label: 'PDF / 파일 선택',
    desc: 'PDF 또는 이미지 파일을 선택하세요',
  },
] as const;

export default function InBodyUploadScreen() {
  const nav = useNavigation<Nav>();
  const [busy, setBusy] = useState(false);

  const upload = async (file: UploadFile) => {
    setBusy(true);
    try {
      const res = await api.uploadInBody(file);
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

  const handleOption = async (id: (typeof OPTIONS)[number]['id']) => {
    if (id === 'camera') {
      const perm = await ImagePicker.requestCameraPermissionsAsync();
      if (!perm.granted) {
        Alert.alert('권한 필요', '카메라 접근 권한이 필요합니다.');
        return;
      }
      const res = await ImagePicker.launchCameraAsync({ quality: 0.9 });
      if (res.canceled) return;
      const asset = res.assets[0];
      await upload({ uri: asset.uri, name: fileNameFromUri(asset.uri), type: asset.mimeType ?? guessMimeFromUri(asset.uri) });
    } else if (id === 'gallery') {
      const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!perm.granted) {
        Alert.alert('권한 필요', '갤러리 접근 권한이 필요합니다.');
        return;
      }
      const res = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.9 });
      if (res.canceled) return;
      const asset = res.assets[0];
      await upload({ uri: asset.uri, name: fileNameFromUri(asset.uri), type: asset.mimeType ?? guessMimeFromUri(asset.uri) });
    } else {
      const res = await DocumentPicker.getDocumentAsync({ type: ['application/pdf', 'image/*'], copyToCacheDirectory: true });
      if (res.canceled) return;
      const asset = res.assets[0];
      await upload({ uri: asset.uri, name: asset.name, type: asset.mimeType ?? guessMimeFromUri(asset.name) });
    }
  };

  if (busy) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#4f6ef7" />
        <Text style={styles.loadingTitle}>AI가 분석 중입니다</Text>
        <Text style={styles.loadingDesc}>인바디 수치를 자동으로 추출하고 있어요{'\n'}약 5~10초 소요됩니다</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.topSection}>
        <Text style={styles.badge}>AI 자동 추출</Text>
        <Text style={styles.title}>인바디 결과지{'\n'}업로드</Text>
        <Text style={styles.desc}>
          사진 촬영 또는 파일 선택 시{'\n'}AI가 수치를 자동으로 읽어드립니다
        </Text>
      </View>

      <View style={styles.options}>
        {OPTIONS.map((opt) => (
          <TouchableOpacity
            key={opt.id}
            style={styles.optionCard}
            onPress={() => handleOption(opt.id)}
            activeOpacity={0.75}
          >
            <Text style={styles.optionIcon}>{opt.icon}</Text>
            <View style={styles.optionText}>
              <Text style={styles.optionLabel}>{opt.label}</Text>
              <Text style={styles.optionDesc}>{opt.desc}</Text>
            </View>
            <Text style={styles.optionArrow}>›</Text>
          </TouchableOpacity>
        ))}
      </View>

      <Text style={styles.footer}>
        지원 형식: JPG · PNG · HEIC · PDF
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f0f4f8',
    padding: 24,
  },
  topSection: {
    marginTop: 8,
    marginBottom: 32,
  },
  badge: {
    alignSelf: 'flex-start',
    backgroundColor: '#e8edff',
    color: '#4f6ef7',
    fontSize: 12,
    fontWeight: '600',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 20,
    marginBottom: 12,
    overflow: 'hidden',
  },
  title: {
    fontSize: 28,
    fontWeight: '800',
    color: '#1a1a2e',
    lineHeight: 36,
    marginBottom: 10,
  },
  desc: {
    fontSize: 14,
    color: '#666',
    lineHeight: 22,
  },
  options: {
    gap: 12,
  },
  optionCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 18,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 6,
    elevation: 2,
  },
  optionIcon: {
    fontSize: 28,
    marginRight: 14,
  },
  optionText: {
    flex: 1,
  },
  optionLabel: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1a1a2e',
    marginBottom: 3,
  },
  optionDesc: {
    fontSize: 13,
    color: '#888',
  },
  optionArrow: {
    fontSize: 22,
    color: '#ccc',
    fontWeight: '300',
  },
  footer: {
    textAlign: 'center',
    color: '#aaa',
    fontSize: 12,
    marginTop: 28,
  },
  loadingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f0f4f8',
    gap: 16,
  },
  loadingTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1a1a2e',
  },
  loadingDesc: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    lineHeight: 22,
  },
});
