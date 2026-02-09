import threading
import uuid
from datetime import datetime
from enum import Enum

class JobState(Enum):
    """Job lifecycle states"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class JobManager:
    """
    Thread-safe job manager for background ML processing.
    
    Why threading instead of Celery/Redis/RQ?
    - Demo stability: No external services required
    - Memory efficiency: Models loaded once, shared across threads
    - Simplicity: Single process, easy to explain and debug
    - Deployment: Works on shared hosting, local development, small VPS
    
    Thread safety: All access to shared state (jobs dict) is protected by locks.
    """
    
    def __init__(self):
        self.jobs = {}  # {job_id: {state, results, error, progress}}
        self.lock = threading.Lock()  # Protect concurrent access
    
    def create_job(self):
        """Create a new job and return its ID"""
        job_id = str(uuid.uuid4())
        with self.lock:
            self.jobs[job_id] = {
                'state': JobState.QUEUED.value,
                'created_at': datetime.now().isoformat(),
                'progress': 0,
                'status_message': 'Job queued, waiting to start',
                'results': None,
                'error': None
            }
        return job_id
    
    def get_job(self, job_id):
        """Get job status (thread-safe)"""
        with self.lock:
            return self.jobs.get(job_id)
    
    def update_progress(self, job_id, progress, status_message=""):
        """Update job progress (0-100)"""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['progress'] = progress
                if status_message:
                    self.jobs[job_id]['status_message'] = status_message
    
    def mark_running(self, job_id):
        """Mark job as running"""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['state'] = JobState.RUNNING.value
                self.jobs[job_id]['status_message'] = 'Processing image...'
    
    def mark_completed(self, job_id, results):
        """Mark job as completed with results"""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['state'] = JobState.COMPLETED.value
                self.jobs[job_id]['results'] = results
                self.jobs[job_id]['progress'] = 100
                self.jobs[job_id]['status_message'] = 'Analysis complete'
    
    def mark_failed(self, job_id, error_message):
        """Mark job as failed with error"""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['state'] = JobState.FAILED.value
                self.jobs[job_id]['error'] = str(error_message)
                self.jobs[job_id]['status_message'] = 'Analysis failed'
    
    def cleanup_job(self, job_id):
        """Remove old job from memory (optional, for memory management)"""
        with self.lock:
            if job_id in self.jobs:
                del self.jobs[job_id]

# Global instance - shared across all Flask requests
job_manager = JobManager()
