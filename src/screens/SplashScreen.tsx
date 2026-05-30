import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

export default function SplashScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Health{'\n'}First</Text>
      <Text style={styles.subtitle}>AI HEALTH RISK ASSESSMENT</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#3D5AFE',
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 42,
    fontWeight: '300',
    color: '#ffffff',
    letterSpacing: 1,
    lineHeight: 52,
    marginBottom: 14,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 12,
    fontWeight: '700',
    color: '#ffffff',
    letterSpacing: 2,
  },
});
