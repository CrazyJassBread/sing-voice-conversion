from huggingface_hub import snapshot_download


if __name__ == "__main__":
    model_path = snapshot_download(
        repo_id="Pur1zumu/RIFT-SVC-modules",
        local_dir='pretrained',
        local_dir_use_symlinks=False,  # Don't use symlinks
        local_files_only=False,        # Allow downloading new files
        ignore_patterns=["*.git*"],    # Ignore git-related files
        resume_download=True           # Resume interrupted downloads
    )
