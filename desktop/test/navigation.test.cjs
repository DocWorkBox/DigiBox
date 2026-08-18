'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  classifyNavigation,
  isMediaPermissionAllowed,
} = require('../lib/navigation.cjs');

test('navigation stays inside the exact AVTR loopback origin', () => {
  assert.equal(classifyNavigation('http://127.0.0.1:7860/settings'), 'app');
  assert.equal(classifyNavigation('http://localhost:7860/'), 'deny');
  assert.equal(classifyNavigation('http://127.0.0.1:8000/docs'), 'deny');
  assert.equal(classifyNavigation('file:///C:/Windows/win.ini'), 'deny');
});

test('only allowlisted HTTPS destinations open in the system browser', () => {
  assert.equal(classifyNavigation('https://platform.qianwenai.com/'), 'external');
  assert.equal(classifyNavigation('https://github.com/avaturn-live/avtr-1'), 'external');
  assert.equal(classifyNavigation('https://attacker.invalid/'), 'deny');
  assert.equal(classifyNavigation('javascript:alert(1)'), 'deny');
});

test('microphone permission is limited to media on the exact app origin', () => {
  assert.equal(isMediaPermissionAllowed('media', 'http://127.0.0.1:7860/'), true);
  assert.equal(isMediaPermissionAllowed('media', 'http://localhost:7860/'), false);
  assert.equal(isMediaPermissionAllowed('notifications', 'http://127.0.0.1:7860/'), false);
});
