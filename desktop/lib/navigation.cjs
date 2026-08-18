'use strict';

const APP_ORIGIN = 'http://127.0.0.1:7860';
const EXTERNAL_HOSTS = new Set([
  'github.com',
  'huggingface.co',
  'platform.qianwenai.com',
  'platform.minimaxi.com',
]);

function classifyNavigation(value, options = {}) {
  let url;
  try {
    url = new URL(value);
  } catch {
    return 'deny';
  }
  const appOrigin = options.appOrigin || APP_ORIGIN;
  if (url.origin === appOrigin) return 'app';
  const hosts = options.externalHosts || EXTERNAL_HOSTS;
  if (url.protocol === 'https:' && hosts.has(url.hostname.toLowerCase())) return 'external';
  return 'deny';
}

function isMediaPermissionAllowed(permission, requestingUrl, options = {}) {
  if (permission !== 'media') return false;
  try {
    return new URL(requestingUrl).origin === (options.appOrigin || APP_ORIGIN);
  } catch {
    return false;
  }
}

module.exports = {
  APP_ORIGIN,
  EXTERNAL_HOSTS,
  classifyNavigation,
  isMediaPermissionAllowed,
};
