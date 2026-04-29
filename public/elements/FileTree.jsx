// VS Code-style file tree for the indexed knowledge base.
//
// Mounted by Python via cl.CustomElement(name="FileTree", props={"tree": {...}}).
// Globally injected by Chainlit:
//   - props          (data from Python)
//   - callAction({name, payload})  →  triggers a @cl.action_callback in app.py
//   - updateElement(nextProps)
//   - sendUserMessage(message, command?)
//   - deleteElement()
//
// shadcn/ui (Accordion, Button, Badge) and lucide-react are pre-bundled.

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Folder,
  FolderOpen,
  FileText,
  FileType,
  File as FileIcon,
  RefreshCw,
} from "lucide-react"

const FILE_ICONS = {
  pdf: FileType,
  md: FileText,
  txt: FileText,
  html: FileText,
}

function FileRow({ meta, onOpen }) {
  const Icon = FILE_ICONS[meta.file_type] || FileIcon
  const isPending = meta.status === "pending"
  return (
    <Button
      key={meta.source_path}
      variant="ghost"
      size="sm"
      className="w-full justify-start text-xs h-7 font-normal"
      onClick={() => onOpen(meta.source_path, meta.file_name)}
    >
      <Icon className="h-3.5 w-3.5 mr-1.5 text-muted-foreground shrink-0" />
      <span className="truncate">{meta.file_name}</span>
      {meta.chunk_count > 0 && (
        <span className="ml-auto text-muted-foreground text-xs shrink-0">
          {meta.chunk_count}
        </span>
      )}
      {isPending && (
        <span className="ml-1 text-yellow-600 text-xs shrink-0">⚠</span>
      )}
    </Button>
  )
}

function FolderNode({ name, node, depth, onOpen }) {
  const files = node._files || []
  const subfolders = Object.keys(node).filter((k) => k !== "_files").sort()
  const allKeys = [...subfolders.map((s) => `${name}/${s}`), ...files.map((f) => f.source_path)]
  const pendingHere =
    files.filter((f) => f.status === "pending").length +
    subfolders.reduce((acc, k) => acc + countPending(node[k]), 0)

  return (
    <AccordionItem value={name} className="border-b-0">
      <AccordionTrigger
        className="text-sm hover:no-underline"
        style={{ paddingTop: "4px", paddingBottom: "4px", paddingLeft: `${depth * 12}px` }}
      >
        <span className="flex items-center gap-1.5">
          <Folder className="h-3.5 w-3.5 text-blue-500 shrink-0" />
          <span className="font-medium text-xs">{name}</span>
          {pendingHere > 0 && (
            <Badge variant="outline" className="text-xs px-1 py-0 text-yellow-600 border-yellow-400">
              ⚠ {pendingHere}
            </Badge>
          )}
        </span>
      </AccordionTrigger>
      <AccordionContent className="pb-0">
        {subfolders.length > 0 && (
          <Accordion type="multiple" defaultValue={subfolders.map((s) => `${name}/${s}`)}>
            {subfolders.map((sub) => (
              <FolderNode
                key={`${name}/${sub}`}
                name={`${name}/${sub}`}
                node={node[sub]}
                depth={depth + 1}
                onOpen={onOpen}
              />
            ))}
          </Accordion>
        )}
        {files.length > 0 && (
          <div
            className="flex flex-col gap-0.5"
            style={{ paddingLeft: `${(depth + 1) * 12}px` }}
          >
            {files.map((f) => (
              <FileRow key={f.source_path} meta={f} onOpen={onOpen} />
            ))}
          </div>
        )}
      </AccordionContent>
    </AccordionItem>
  )
}

function countPending(node) {
  if (!node || typeof node !== "object") return 0
  const files = node._files || []
  const fromFiles = files.filter((f) => f.status === "pending").length
  const fromSubs = Object.keys(node)
    .filter((k) => k !== "_files")
    .reduce((acc, k) => acc + countPending(node[k]), 0)
  return fromFiles + fromSubs
}

function countFiles(node) {
  if (!node || typeof node !== "object") return 0
  const files = (node._files || []).length
  const fromSubs = Object.keys(node)
    .filter((k) => k !== "_files")
    .reduce((acc, k) => acc + countFiles(node[k]), 0)
  return files + fromSubs
}

function countChunks(node) {
  if (!node || typeof node !== "object") return 0
  const chunks = (node._files || []).reduce((a, f) => a + (f.chunk_count || 0), 0)
  const fromSubs = Object.keys(node)
    .filter((k) => k !== "_files")
    .reduce((acc, k) => acc + countChunks(node[k]), 0)
  return chunks + fromSubs
}

export default function FileTree() {
  const tree = (props && props.tree) || {}
  const topFolders = Object.keys(tree).sort()
  const totalFiles = countFiles(tree)
  const totalChunks = countChunks(tree)
  const totalPending = countPending(tree)

  const handleOpen = (sp, fn) => {
    callAction({ name: "open_file", payload: { source_path: sp, file_name: fn } })
  }

  const handleRefresh = () => {
    callAction({ name: "refresh_file_tree", payload: {} })
  }

  return (
    <div className="border rounded-lg p-3 bg-card text-card-foreground w-full shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between pb-2 mb-1 border-b">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <FolderOpen className="h-4 w-4 text-primary" />
          <span>File system</span>
          <span className="text-xs text-muted-foreground font-normal">
            {totalFiles} files · {totalChunks} chunks
          </span>
          {totalPending > 0 && (
            <Badge variant="destructive" className="text-xs">
              ⚠ {totalPending} pending
            </Badge>
          )}
        </div>
        <Button
          size="icon"
          variant="ghost"
          className="h-7 w-7"
          onClick={handleRefresh}
          title="Refresh from Azure"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </Button>
      </div>

      {/* Tree */}
      <Accordion type="multiple" defaultValue={topFolders}>
        {topFolders.map((folder) => (
          <FolderNode
            key={folder}
            name={folder}
            node={tree[folder]}
            depth={0}
            onOpen={handleOpen}
          />
        ))}
      </Accordion>

      <div className="mt-2 pt-2 border-t text-xs text-muted-foreground">
        Click a file to open it in the side viewer.
      </div>
    </div>
  )
}
