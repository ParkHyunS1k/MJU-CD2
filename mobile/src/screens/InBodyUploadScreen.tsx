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

  const pickFromGallery = async () => {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('권한 필요', '갤러리 접근 권한이 필요합니다.');
      return;
    }
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.9,
    });
    if (res.canceled) return;
    const asset = res.assets[0];
    await upload({
      uri: asset.uri,
      name: fileNameFromUri(asset.uri),
      type: asset.mimeType ?? guessMimeFromUri(asset.uri),
    });
  };

  const takePhoto = async () => {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('권한 필요', '카메라 접근 권한이 필요합니다.');
      return;
    }
    const res = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.9,
    });
    if (res.canceled) return;
    const asset = res.assets[0];
    await upload({
      uri: asset.uri,
      name: fileNameFromUri(asset.uri),
      type: asset.mimeType ?? guessMimeFromUri(asset.uri),
    });
  };

  const pickPdf = async () => {
    const res = await DocumentPicker.getDocumentAsync({
      type: ['application/pdf', 'image/*'],
      copyToCacheDirectory: true,
    });
    if (res.canceled) return;
    const asset = res.assets[0];
    await upload({
      uri: asset.uri,
      name: asset.name,
      type: asset.mimeType ?? guessMimeFromUri(asset.name),
    });
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>인바디 결과지 업로드</Text>
      <Text style={styles.subtitle}>
        사진을 찍거나 파일(이미지/PDF)을 선택하면 AI가 수치를 자동 추출합니다.
      </Text>

      {busy ? (
        <View style={styles.busyBox}>
          <ActivityIndicator size="large" />
          <Text style={styles.busyText}>분석 중... 약 5~10초 소요</Text>
        </View>
      ) : (
        <View style={{ gap: 12 }}>
          <TouchableOpacity style={styles.button} onPress={takePhoto}>
            <Text style={styles.buttonText}>📷 카메라로 촬영</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={pickFromGallery}>
            <Text style={styles.buttonText}>🖼  갤러리에서 선택</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={pickPdf}>
            <Text style={styles.buttonText}>📄 PDF/파일 선택</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, gap: 24, backgroundColor: '#fff' },
  title: { fontSize: 22, fontWeight: '700' },
  subtitle: { fontSize: 14, color: '#666' },
  button: {
    backgroundColor: '#1f6feb',
    padding: 16,
    borderRadius: 10,
    alignItems: 'center',
  },
  buttonText: { color: '#fff', fontWeight: '600', fontSize: 16 },
  busyBox: { alignItems: 'center', gap: 12, marginTop: 40 },
  busyText: { color: '#444' },
});
