from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
from .models import Station, Train, TrainTrip, TamperMachine, TamperOperation, Report, ReportBatch, Route 
from rest_framework import viewsets, permissions
from .serializers import ReportUpdateSerializer, UserSerializer, StationSerializer, TrainSerializer, TamperOperationSerializer, TamperMachineSerializer
from .serializers import ReportBatchCreateSerializer, TrainTripSerializer, ReportBatchSerializer, ReportSerializer, RouteSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAdminUser]

class StationViewSet(viewsets.ModelViewSet):
    queryset = Station.objects.all()
    serializer_class = StationSerializer
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [permissions.IsAdminUser()]
        
        return [permissions.IsAuthenticated()]
    
class TrainViewSet(viewsets.ModelViewSet):
    queryset = Train.objects.all()
    serializer_class = TrainSerializer
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [permissions.IsAdminUser()]
        
        return [permissions.IsAuthenticated()]

    
class TrainTripViewSet(viewsets.ModelViewSet):
    queryset = TrainTrip.objects.all()
    serializer_class = TrainTripSerializer
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [permissions.IsAdminUser()]
        
        return [permissions.IsAuthenticated()]
    
class RouteViewSet(viewsets.ModelViewSet):
    queryset = Route.objects.all()
    serializer_class = RouteSerializer
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [permissions.IsAdminUser()]
        
        return [permissions.IsAuthenticated()]
    
class TrainTripViewSet(viewsets.ModelViewSet):
    queryset = TrainTrip.objects.all()
    serializer_class = TrainTripSerializer
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [permissions.IsAdminUser()]
        
        return [permissions.IsAuthenticated()]
   
class TamperMachineViewSet(viewsets.ModelViewSet):
    queryset = TamperMachine.objects.all()
    serializer_class = TamperMachineSerializer
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [permissions.IsAdminUser()]
        
        return [permissions.IsAuthenticated()]
    
class TamperOperationViewSet(viewsets.ModelViewSet):
    queryset = TamperOperation.objects.all()
    serializer_class = TamperOperationSerializer
    def get_permissions(self):
        if self.action == 'destroy':
            return [permissions.IsAdminUser()]
        
        return [permissions.IsAuthenticated()]

class ReportBatchAPI(APIView):

    def get_permissions(self):
        if self.request.method == "POST":
            return [permissions.IsAdminUser()]          # Solo admins pueden hacer POST
        elif self.request.method == "GET":
            return [permissions.IsAuthenticated()]      # Cualquier usuario autenticado
        return super().get_permissions()

    def post(self, request):
        serializer = ReportBatchCreateSerializer(data=request.data)
        if serializer.is_valid():
            batch = serializer.save()
            return Response({
                "message": "Report batch created successfully",
                "batch_id": batch.id
            }, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        batch_id = request.query_params.get("id")

        if batch_id:
            batch = get_object_or_404(ReportBatch, id=batch_id)
            serializer = ReportBatchSerializer(batch)
            return Response(serializer.data, status=status.HTTP_200_OK)

        batches = ReportBatch.objects.all()
        serializer = ReportBatchSerializer(batches, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    
class ReportUpdateAPI(APIView):
    permission_classes = [permissions.IsAdminUser]  # opcional

    def patch(self, request, report_id):
        report = get_object_or_404(Report, id=report_id)

        serializer = ReportUpdateSerializer(
            report,
            data=request.data,
            partial=True  # permite actualizar uno o ambos campos
        )

        if serializer.is_valid():
            serializer.save()
            return Response(
                {"message": "Report updated successfully", "report": serializer.data},
                status=status.HTTP_200_OK
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    # Permitir también PUT si quieres
    def put(self, request, report_id):
        return self.patch(request, report_id)