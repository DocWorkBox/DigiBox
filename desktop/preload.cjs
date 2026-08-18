'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('avtrDesktop', Object.freeze({
  getState: () => ipcRenderer.invoke('desktop:get-state'),
  retry: () => ipcRenderer.invoke('desktop:retry'),
  selectRuntime: () => ipcRenderer.invoke('desktop:select-runtime'),
  openLogs: () => ipcRenderer.invoke('desktop:open-logs'),
  onState: (callback) => {
    if (typeof callback !== 'function') throw new TypeError('callback must be a function');
    const listener = (_event, state) => callback(state);
    ipcRenderer.on('desktop:state', listener);
    return () => ipcRenderer.removeListener('desktop:state', listener);
  },
}));
