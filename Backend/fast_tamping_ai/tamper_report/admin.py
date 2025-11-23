from django.contrib import admin
from .models import Station, Train, Route, TrainTrip, Report
from .models import ReportBatch, TamperMachine, TamperOperation


@admin.register(Station)
class StationAdmin(admin.ModelAdmin):
    list_display = ("name", "city", "latitude", "longitude")
    search_fields = ("name", "city")


@admin.register(Train)
class TrainAdmin(admin.ModelAdmin):
    list_display = ("model", "train_type", "manufacturer", "year_built", "max_speed", "weight")
    search_fields = ("name", "model", "manufacturer")
    list_filter = ("train_type", "manufacturer", "year_built")


@admin.register(Route)
class RouteAdmin(admin.ModelAdmin):
    list_display = ("route_code", "origin", "destination")
    search_fields = ("route_code",)
    list_filter = ("origin", "destination")


@admin.register(TrainTrip)
class TrainTripAdmin(admin.ModelAdmin):
    list_display = (
        "train",
        "route",
        "cargo_weight",
        "average_speed",
        "departure_time",
        "arrival_time",
    )
    list_filter = ("train", "route")
    search_fields = ("train__name", "route__route_code")


@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    list_display = (
        "report_batch",
        "lift_left_mm",
        "lift_right_mm",
        "adjustment_left_mm",
        "adjustment_right_mm",
        "state",
        "date_reported",
        "date_fixed",
        "start_latitude",
        "start_longitude",
        "end_latitude",
        "end_longitude",
    )
    list_filter = ("state", "report_batch__train_trip__route", "date_reported")
    search_fields = ("report_batch__train_trip__train__name", "report_batch__train_trip__route__route_code")


@admin.register(ReportBatch)
class ReportBatchAdmin(admin.ModelAdmin):
    list_display = ("id", "train_trip", "created_at")
    search_fields = ("train_trip__id", "train_trip__route__route_code")
    list_filter = ("created_at",)
    ordering = ("-created_at",)


@admin.register(TamperMachine)
class TamperMachineAdmin(admin.ModelAdmin):
    list_display = ("id", "model", "manufacturer", "tamper_type", "year_built")
    search_fields = ("model", "manufacturer")
    list_filter = ("tamper_type", "manufacturer", "year_built")
    ordering = ("id",)


@admin.register(TamperOperation)
class TamperOperationAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "tamper",
        "report_batch",
        "start_time",
        "end_time",
        "operator",
    )
    search_fields = (
        "tamper__model",
        "operator__username",
        "report_batch__train_trip__route__route_code",
    )
    list_filter = ("start_time", "operator")
    ordering = ("-start_time",)

