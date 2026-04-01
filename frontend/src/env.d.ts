/// <reference types="vite/client" />

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}

// FileSystem API types for drag-and-drop folder support
interface FileSystemEntry {
  readonly isFile: boolean
  readonly isDirectory: boolean
  readonly name: string
}

interface FileSystemFileEntry extends FileSystemEntry {
  file(success: (file: File) => void, error?: () => void): void
}

interface FileSystemDirectoryEntry extends FileSystemEntry {
  createReader(): FileSystemDirectoryReader
}

interface FileSystemDirectoryReader {
  readEntries(success: (entries: FileSystemEntry[]) => void, error?: () => void): void
}

interface DataTransferItem {
  webkitGetAsEntry?(): FileSystemEntry | null
}
