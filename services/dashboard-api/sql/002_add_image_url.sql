-- Add image_url column for cropped image URL after upload (run on existing DBs)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'video_frame_records' AND column_name = 'image_url'
  ) THEN
    ALTER TABLE video_frame_records ADD COLUMN image_url TEXT;
  END IF;
END $$;
