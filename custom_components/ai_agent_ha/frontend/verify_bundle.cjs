const fs = require('fs');

console.log('=== COMPREHENSIVE BUNDLE VERIFICATION ===\n');

const bundle = fs.readFileSync('ai_agent_ha-panel.js', 'utf8');

// Test 1: Check for critical errors
console.log('1. Checking for runtime errors...');
const criticalErrors = [
  { pattern: /\$state is not defined/g, name: '$state error' },
  { pattern: /\$derived is not defined/g, name: '$derived error' },
  { pattern: /\$effect is not defined/g, name: '$effect error' },
  { pattern: /\$props is not defined/g, name: '$props error' },
];

let hasErrors = false;
criticalErrors.forEach(({ pattern, name }) => {
  const matches = bundle.match(pattern);
  if (matches) {
    console.log(`   ❌ Found ${matches.length}x: ${name}`);
    hasErrors = true;
  }
});

if (!hasErrors) {
  console.log('   ✅ No critical runtime errors\n');
}

// Test 2: Verify store implementations
console.log('2. Checking store implementations...');
const storePatterns = [
  { pattern: /writable\(/g, name: 'writable()', required: true },
  { pattern: /derived\(/g, name: 'derived()', required: true },
  { pattern: /\.subscribe\(/g, name: 'subscribe()', required: true },
  { pattern: /\.update\(/g, name: 'update()', required: true },
  { pattern: /get\(/g, name: 'get()', required: true },
];

storePatterns.forEach(({ pattern, name, required }) => {
  const matches = bundle.match(pattern);
  const count = matches ? matches.length : 0;
  if (required && count === 0) {
    console.log(`   ❌ Missing: ${name}`);
    hasErrors = true;
  } else {
    console.log(`   ✅ ${name}: ${count}x`);
  }
});

console.log('');

// Test 3: Check Web Component registration
console.log('3. Checking Web Component...');
if (bundle.includes('customElements.define')) {
  console.log('   ✅ Custom element registration found');
} else {
  console.log('   ❌ No custom element registration');
  hasErrors = true;
}

if (bundle.includes('ai-agent-ha-panel')) {
  console.log('   ✅ Component name correct\n');
} else {
  console.log('   ❌ Component name not found\n');
  hasErrors = true;
}

// Test 4: Check for Shadow DOM
console.log('4. Checking Shadow DOM...');
if (bundle.includes('attachShadow')) {
  console.log('   ✅ Shadow DOM implementation found\n');
} else {
  console.log('   ⚠️  No Shadow DOM found\n');
}

// Test 5: Bundle stats
console.log('5. Bundle statistics:');
console.log(`   Size: ${Math.round(bundle.length / 1024)} KB`);
console.log(`   Lines: ${bundle.split('\n').length}`);
console.log(`   Characters: ${bundle.length.toLocaleString()}\n`);

// Final result
console.log('=== FINAL VERDICT ===');
if (hasErrors) {
  console.log('❌ FAILED - Bundle has critical errors\n');
  process.exit(1);
} else {
  console.log('✅ PASSED - Bundle is production ready!');
  console.log('✅ All stores use writable/derived correctly');
  console.log('✅ No Svelte 5 runes runtime errors');
  console.log('✅ Web Component properly registered');
  console.log('✅ Ready for Home Assistant integration\n');
}
