from django.contrib.auth.models import User
from .models import Report, ReportBatch, Station, TamperMachine, TamperOperation, Train, Route, TrainTrip
from rest_framework import serializers

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['id', 'username', 'password', 'first_name', 'last_name']

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User(**validated_data)
        user.set_password(password)
        user.save()
        return user

    def update(self, instance, validated_data):
        password = validated_data.pop('password', None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if password:
            instance.set_password(password)
        instance.save()
        return instance
    

class StationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Station
        fields = ['id', 'name', 'city', 'latitude', 'longitude']


class TrainSerializer(serializers.ModelSerializer):
    class Meta:
        model = Train
        fields = ['id','model','weight','max_speed','train_type','manufacturer','year_built']

class RouteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Route
        fields = ['id', 'route_code', 'origin', 'destination']


class TrainTripSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainTrip
        fields = [
            'id', 'train', 'route',
            'cargo_weight', 'average_speed',
            'departure_time', 'arrival_time'
        ]

class ReportCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Report
        fields = [
            "start_latitude", "start_longitude",
            "end_latitude", "end_longitude",
            "lift_mm",
            "adjustment_left_mm", "adjustment_right_mm",
            "state",
        ]
        extra_kwargs = {
            "state": {"required": False},  
        }

class ReportBatchCreateSerializer(serializers.Serializer):
    train_trip = serializers.IntegerField()
    reports = ReportCreateSerializer(many=True)

    def create(self, validated_data):
        train_trip_id = validated_data["train_trip"]
        reports_data = validated_data["reports"]

        train_trip = TrainTrip.objects.get(id=train_trip_id)

        batch = ReportBatch.objects.create(train_trip=train_trip)

        report_objects = [
            Report(
                report_batch=batch,
                **report_data
            )
            for report_data in reports_data
        ]
        Report.objects.bulk_create(report_objects)

        return batch

class ReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Report
        fields = [
            "id",
            "start_latitude", "start_longitude",
            "end_latitude", "end_longitude",
            "lift_mm",
            "adjustment_left_mm", "adjustment_right_mm",
            "state",
            "date_reported",
            "date_fixed",
        ]

class ReportBatchSerializer(serializers.ModelSerializer):
    train_trip = TrainTripSerializer(read_only=True)
    reports = ReportSerializer(source="report_set", many=True)

    class Meta:
        model = ReportBatch
        fields = [
            "id",
            "created_at",
            "train_trip",
            "reports",
        ]


class ReportUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Report
        fields = ["state", "date_fixed"]

        extra_kwargs = {
            "state": {"required": True},
            "date_fixed": {"required": False},
        }



class TamperMachineSerializer(serializers.ModelSerializer):
    class Meta:
        model = TamperMachine
        fields = '__all__'

class TamperOperationSerializer(serializers.ModelSerializer):
    class Meta:
        model = TamperOperation
        fields = [
            'id', 'tamper', 'report_batch',
            'start_time', 'end_time', 'operator'
        ]



