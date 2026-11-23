import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';

import { PRIVACY_POLICY, TERMS_OF_SERVICE, type PolicyBlock } from '../content/policies';
import type { SettingsStackParamList } from '../navigation/types';

type Props = NativeStackScreenProps<SettingsStackParamList, 'Policy'>;

const DOCS = { privacy: PRIVACY_POLICY, terms: TERMS_OF_SERVICE };

export default function PolicyScreen({ route }: Props) {
  const doc = DOCS[route.params.kind];

  return (
    <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
      <Text style={styles.effective}>{doc.effectiveDate}</Text>
      <Text style={styles.intro}>{doc.intro}</Text>

      {doc.sections.map((section) => (
        <View key={section.title} style={styles.section}>
          <Text style={styles.sectionTitle}>{section.title}</Text>
          {section.blocks.map((block, i) => (
            <Block key={i} block={block} />
          ))}
        </View>
      ))}
    </ScrollView>
  );
}

function Block({ block }: { block: PolicyBlock }) {
  if (typeof block === 'string') {
    return <Text style={styles.paragraph}>{block}</Text>;
  }
  return (
    <View style={styles.list}>
      {block.map((item) => (
        <View key={item} style={styles.listItem}>
          <Text style={styles.bullet}>•</Text>
          <Text style={styles.listText}>{item}</Text>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  scroll: { flex: 1, backgroundColor: '#FFFFFF' },
  content: { padding: 20, paddingBottom: 48 },

  effective: { fontSize: 12, color: '#999999', marginBottom: 8 },
  intro: { fontSize: 14, lineHeight: 22, color: '#333333', marginBottom: 8 },

  section: { marginTop: 20 },
  sectionTitle: { fontSize: 15, fontWeight: '700', color: '#1A1A1A', marginBottom: 8 },
  paragraph: { fontSize: 14, lineHeight: 22, color: '#444444', marginBottom: 8 },

  list: { marginBottom: 8 },
  listItem: { flexDirection: 'row', marginBottom: 4 },
  bullet: { fontSize: 14, lineHeight: 22, color: '#00A38D', width: 14 },
  listText: { flex: 1, fontSize: 14, lineHeight: 22, color: '#444444' },
});
