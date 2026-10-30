import React, { useState } from 'react';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import {
  Alert,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as DocumentPicker from 'expo-document-picker';
import * as ImagePicker from 'expo-image-picker';

import { api, UploadFile } from '../lib/api';
import type { InBodyStackParamList } from '../navigation/types';

type Nav = NativeStackNavigationProp<InBodyStackParamList, 'InBodyUpload'>;

const TEAL = '#00C2A8';
const MAX_BYTES = 15 * 1024 * 1024; // 서버 제한과 같음

// 사진 보관함·카메라: 아이폰의 HEIC 사진을 JPG로 바꿔 받는다(서버·AI 호환)
const PHOTO_OPTIONS: ImagePicker.ImagePickerOptions = {
  mediaTypes: ['images'],
  quality: 0.85,
  preferredAssetRepresentationMode:
    ImagePicker.UIImagePickerPreferredAssetRepresentationMode.Compatible,
};

function guessMimeFromUri(uri: string): string {
  const lower = uri.toLowerCase();
  if (lower.endsWith('.png')) return 'image/png';
  if (lower.endsWith('.heic') || lower.endsWith('.heif')) return 'image/heic';
  if (lower.endsWith('.webp')) return 'image/webp';
  if (lower.endsWith('.pdf')) return 'application/pdf';
  return 'image/jpeg';
}

export default function InBodyUploadScreen() {
  const nav = useNavigation<Nav>();
  const [busy, setBusy] = useState(false);
  const [selectedFile, setSelectedFile] = useState<UploadFile | null>(null);

  // 고른 파일을 서버가 받는 형식으로 정리해서 저장한다
  const select = (file: UploadFile, size?: number | null) => {
    if (size && size > MAX_BYTES) {
      Alert.alert('파일이 너무 커요', '15MB 이하의 파일을 선택해 주세요.');
      return;
    }
    // 서버는 image/heic만 받으므로 heif도 heic로 보낸다
    const type = file.type === 'image/heif' ? 'image/heic' : file.type;
    setSelectedFile({ ...file, type });
  };

  const fromImageAsset = (asset: ImagePicker.ImagePickerAsset) => {
    const name = asset.fileName ?? `inbody_${Date.now()}.jpg`;
    select(
      { uri: asset.uri, name, type: asset.mimeType ?? guessMimeFromUri(name) },
      asset.fileSize,
    );
  };

  // 사진 보관함: 아이폰은 문서 선택기로는 사진 앱에 접근할 수 없어서 따로 연다
  const pickFromLibrary = async () => {
    const res = await ImagePicker.launchImageLibraryAsync(PHOTO_OPTIONS);
    if (!res.canceled) fromImageAsset(res.assets[0]);
  };

  const takePhoto = async () => {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('카메라 권한이 필요해요', '설정에서 카메라 접근을 허용해 주세요.');
      return;
    }
    const res = await ImagePicker.launchCameraAsync(PHOTO_OPTIONS);
    if (!res.canceled) fromImageAsset(res.assets[0]);
  };

  // 파일: 인바디 앱에서 내보낸 PDF 등
  const pickFile = async () => {
    const res = await DocumentPicker.getDocumentAsync({
      type: ['image/*', 'application/pdf'],
      copyToCacheDirectory: true,
    });
    if (res.canceled) return;
    const asset = res.assets[0];
    select(
      { uri: asset.uri, name: asset.name, type: asset.mimeType ?? guessMimeFromUri(asset.name) },
      asset.size,
    );
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      await pickFromLibrary();
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
    '인바디 측정지 사진 (JPG, PNG, HEIC)',
    '인바디 앱 PDF 출력물',
    '최대 파일 크기: 15MB',
    '결과지 전체가 보이고 글자가 흐리지 않게 찍어 주세요',
  ];
  const isImage = selectedFile?.type.startsWith('image/') ?? false;

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

        <View style={styles.uploadZone}>
          {selectedFile && isImage ? (
            <Image source={{ uri: selectedFile.uri }} style={styles.preview} resizeMode="contain" />
          ) : (
            <Ionicons
              name={selectedFile ? 'document-text-outline' : 'cloud-upload-outline'}
              size={40}
              color={selectedFile ? TEAL : '#CCCCCC'}
            />
          )}
          <Text style={styles.uploadZoneTitle} numberOfLines={1}>
            {selectedFile ? selectedFile.name : '아래에서 결과지를 선택하세요'}
          </Text>
          {selectedFile ? (
            <View style={styles.fileSelectedBadge}>
              <Ionicons name="checkmark-circle" size={14} color={TEAL} />
              <Text style={styles.fileSelectedText}>파일 선택됨 · 다시 고르려면 아래 버튼</Text>
            </View>
          ) : (
            <Text style={styles.uploadZoneSub}>PDF, JPG, PNG, HEIC 파일 지원</Text>
          )}
        </View>

        <View style={styles.sourceRow}>
          {[
            { icon: 'images-outline' as const, label: '사진 보관함', onPress: pickFromLibrary },
            { icon: 'camera-outline' as const, label: '카메라 촬영', onPress: takePhoto },
            { icon: 'folder-open-outline' as const, label: '파일 (PDF)', onPress: pickFile },
          ].map((opt) => (
            <TouchableOpacity
              key={opt.label}
              style={styles.sourceBtn}
              onPress={opt.onPress}
              disabled={busy}
              activeOpacity={0.7}
            >
              <Ionicons name={opt.icon} size={24} color={TEAL} />
              <Text style={styles.sourceText}>{opt.label}</Text>
            </TouchableOpacity>
          ))}
        </View>

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
  preview: { width: '80%', height: 160, borderRadius: 8, backgroundColor: '#F5F5F5' },
  sourceRow: { flexDirection: 'row', gap: 10, marginBottom: 16 },
  sourceBtn: {
    flex: 1,
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#fff',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#B5EDE3',
    paddingVertical: 14,
  },
  sourceText: { fontSize: 12, fontWeight: '600', color: '#333' },
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
