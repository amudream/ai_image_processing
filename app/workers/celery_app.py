from __future__ import annotations

from celery import Celery

from app.core.config import settings

celery_app = Celery("image_factory", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.task_routes = {
    "app.workers.tasks.ingest_*": {"queue": "queue.ingest"},
    "app.workers.tasks.analysis_*": {"queue": "queue.analysis"},
    "app.workers.tasks.generation_*": {"queue": "queue.generate"},
    "app.workers.tasks.qa_*": {"queue": "queue.qa"},
    "app.workers.tasks.retry_*": {"queue": "queue.retry"},
    "app.workers.tasks.librarian_*": {"queue": "queue.librarian"},
    "app.workers.tasks.deadletter_*": {"queue": "queue.deadletter"},
}
celery_app.conf.task_acks_late = True
celery_app.conf.task_reject_on_worker_lost = True
celery_app.conf.worker_prefetch_multiplier = 1
