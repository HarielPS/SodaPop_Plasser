from rest_framework.routers import DefaultRouter
from django.urls import path

from .views import (
    ReportUpdateAPI,
    UserViewSet,
    StationViewSet,
    TrainViewSet,
    RouteViewSet,
    TrainTripViewSet,
    ReportBatchAPI,  
    TamperMachineViewSet,
    TamperOperationViewSet
)

router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'stations', StationViewSet)
router.register(r'trains', TrainViewSet)
router.register(r'routes', RouteViewSet)
router.register(r'train-trips', TrainTripViewSet)
router.register(r'tamper-machines', TamperMachineViewSet)
router.register(r'tamper-operations', TamperOperationViewSet)

urlpatterns = [
    path("report-batches/", ReportBatchAPI.as_view(), name="report-batches"),
    path("reports/<int:report_id>/update/", ReportUpdateAPI.as_view(), name="report-update"),
]

urlpatterns += router.urls
