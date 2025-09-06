"""Example usage of the latency tracking system in a service context."""

from datetime import datetime, timezone
from src.utils.io import LatencyTimer, LatencyAggregator, MetricsWriter, get_hour_bucket
from src.config.schemas import HourlyMetrics, PrecisionAtKSnapshot


class TrendPredictionService:
    """Example service showing how to integrate latency tracking."""
    
    def __init__(self):
        self.latency_aggregator = LatencyAggregator()
        self.metrics_writer = MetricsWriter()
        self.current_hour_bucket = None
    
    def process_event(self, event_data):
        """Process a single event with full latency tracking."""
        
        with LatencyTimer() as timer:
            # Ingest stage
            timer.start_stage('ingest')
            ingested_event = self._ingest_event(event_data)
            timer.end_stage('ingest')
            
            # Preprocess stage
            timer.start_stage('preprocess')
            preprocessed_event = self._preprocess_event(ingested_event)
            timer.end_stage('preprocess')
            
            # Model update and forward pass stage
            timer.start_stage('model_update_forward')
            predictions = self._model_forward(preprocessed_event)
            timer.end_stage('model_update_forward')
            
            # Postprocess stage
            timer.start_stage('postprocess')
            result = self._postprocess_predictions(predictions)
            timer.end_stage('postprocess')
        
        # Add measurement to aggregator
        self.latency_aggregator.add_measurement(
            timer.total_duration_ms,
            timer.get_stage_ms()
        )
        
        # Check if we need to write hourly snapshot
        current_time = datetime.now(timezone.utc)
        hour_bucket = get_hour_bucket(current_time)
        
        if self.current_hour_bucket != hour_bucket:
            if self.current_hour_bucket is not None:
                # Write previous hour's metrics
                self._write_hourly_metrics(self.current_hour_bucket)
            
            self.current_hour_bucket = hour_bucket
            self.latency_aggregator.clear()
        
        return result
    
    def _ingest_event(self, event_data):
        """Simulate event ingestion."""
        # In real implementation, this would parse and validate the event
        return {"parsed": event_data}
    
    def _preprocess_event(self, event):
        """Simulate event preprocessing."""
        # In real implementation, this would extract features, embeddings, etc.
        return {"preprocessed": event}
    
    def _model_forward(self, event):
        """Simulate model forward pass."""
        # In real implementation, this would run the TGN model
        return {"predictions": [0.8, 0.6, 0.4]}
    
    def _postprocess_predictions(self, predictions):
        """Simulate prediction postprocessing."""
        # In real implementation, this would format and filter predictions
        return {"final_predictions": predictions}
    
    def _write_hourly_metrics(self, hour_bucket):
        """Write hourly metrics snapshot."""
        # Get latency summary
        latency_summary = self.latency_aggregator.get_summary()
        
        # In real implementation, you would also calculate precision metrics
        # For demo purposes, using dummy values
        precision_snapshot = PrecisionAtKSnapshot(
            k5=0.85,
            k10=0.78,
            support=100
        )
        
        hourly_metrics = HourlyMetrics(
            precision_at_k=precision_snapshot,
            adaptivity=0.0,
            latency=latency_summary,
            meta={"service": "trend_prediction"}
        )
        
        self.metrics_writer.write_hourly_snapshot(hour_bucket, hourly_metrics)
        
        # Log SLO compliance
        slo_status = self.latency_aggregator.meets_slo()
        print(f"Hour {hour_bucket}: SLO compliance = {slo_status}")
        if not all(slo_status.values()):
            print(f"⚠️  SLO violation detected: {latency_summary}")
    
    def get_current_latency_status(self):
        """Get current latency status for monitoring."""
        summary = self.latency_aggregator.get_summary()
        slo_status = self.latency_aggregator.meets_slo()
        
        return {
            "current_summary": summary,
            "slo_compliance": slo_status,
            "measurement_count": len(self.latency_aggregator.measurements)
        }


# Example usage
if __name__ == "__main__":
    service = TrendPredictionService()
    
    # Simulate processing some events
    for i in range(5):
        event_data = {"event_id": f"evt_{i}", "data": f"test_data_{i}"}
        result = service.process_event(event_data)
        print(f"Processed event {i}: {result}")
    
    # Check current status
    status = service.get_current_latency_status()
    print(f"\nCurrent latency status: {status}")
