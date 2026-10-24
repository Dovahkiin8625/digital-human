import type { ReactNode } from "react";
import { Upload } from "antd";
import { InboxOutlined } from "@ant-design/icons";
import type { UploadFile } from "antd/es/upload/interface";

export interface UploadFieldProps {
  /** The `accept` filter, e.g. "image/*,video/*" or ".zip,application/zip". */
  accept: string;
  dragText: ReactNode;
  hint: ReactNode;
  fileList: UploadFile[];
  onChange: (next: UploadFile[]) => void;
}

/**
 * The drop zone both create branches use.
 *
 * The two branches take genuinely different files -- a photo or clip
 * versus a zipped Cubism project -- but the shell around them is the
 * same control with the same one-file limit and the same never-upload
 * behaviour, so it lives here once.
 */
export default function UploadField({
  accept,
  dragText,
  hint,
  fileList,
  onChange,
}: UploadFieldProps) {
  return (
    <Upload.Dragger
      accept={accept}
      maxCount={1}
      listType="picture"
      beforeUpload={() => false}
      fileList={fileList}
      onChange={({ fileList: next }) => onChange(next)}
    >
      <p className="ant-upload-drag-icon">
        <InboxOutlined />
      </p>
      <p className="ant-upload-drag-text">{dragText}</p>
      <p className="ant-upload-hint">{hint}</p>
    </Upload.Dragger>
  );
}
