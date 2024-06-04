import pypipeec.checkpoint as ckpt

DISABLE_LOG: bool = True
DISABLE_SUSPEND: bool = False
DISABLE_RESUME: bool = False

def pipecheck_suspend_transfer(rank: int):
    r"""PipeCheck suspend transfer function."""
    if not DISABLE_LOG and not DISABLE_SUSPEND:
        print(f"[RANK {rank}] | ===== suspend_transfer =====")
    if not DISABLE_SUSPEND:
        ckpt.suspend_transfer()

def pipecheck_suspend_snapshot(rank: int):
    r"""PipeCheck suspend snapshot function."""
    if not DISABLE_LOG and not DISABLE_SUSPEND:
        print(f"[RANK {rank}] | ===== suspend_snapshot =====")
    if not DISABLE_SUSPEND:
        ckpt.suspend_snapshot()

def pipecheck_resume_transfer(rank: int):
    r"""PipeCheck resume transfer function."""
    if not DISABLE_LOG and not DISABLE_RESUME:
        print(f"[RANK {rank}] | ===== resume_transfer =====")
    if not DISABLE_RESUME:
        ckpt.resume_transfer()

def pipecheck_resume_snapshot():
    r"""PipeCheck resume snapshot function."""
    if not DISABLE_RESUME:
        ckpt.resume_snapshot()
