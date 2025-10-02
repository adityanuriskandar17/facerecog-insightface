# FTL GYM Face Recognition - Docker Setup

## 🚀 Quick Start

### 1. Prerequisites
- Docker & Docker Compose
- At least 4GB RAM
- 10GB free disk space

### 2. Setup
```bash
# Clone and navigate to project
cd /home/aditya-nur-iskandar/Downloads/testing1

# Copy environment file
cp env.example .env

# Edit environment variables
nano .env

# Start all services
docker-compose up -d
```

### 3. Access URLs
- **FTL GYM App**: http://localhost:8080
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601
- **Jaeger**: http://localhost:16686
- **Elasticsearch**: http://localhost:9200

## 📊 Monitoring Stack

### ELK Stack (Logging)
- **Elasticsearch**: Log storage and search
- **Kibana**: Log visualization
- **Logstash**: Log processing
- **Fluentd**: Log aggregation

### Prometheus + Grafana (Metrics)
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **Custom Dashboard**: FTL GYM specific metrics

### Jaeger (Tracing)
- **Distributed Tracing**: Request flow tracking
- **Performance Analysis**: Bottleneck identification

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://postgres:password@postgres:5432/ftl_gym

# Redis
REDIS_URL=redis://redis:6379

# Monitoring
ELASTICSEARCH_URL=http://elasticsearch:9200
PROMETHEUS_URL=http://prometheus:9090
JAEGER_AGENT_HOST=jaeger
JAEGER_AGENT_PORT=6831
```

### Custom Metrics
```python
# Face recognition metrics
face_recognition_requests = Counter('face_recognition_requests_total', 'Total requests')
face_recognition_duration = Histogram('face_recognition_duration_seconds', 'Duration')
active_users = Gauge('active_users_current', 'Active users')
```

## 📈 Grafana Dashboard

### Pre-configured Panels
1. **Active Users**: Real-time user count
2. **Success Rate**: Face recognition accuracy
3. **Request Rate**: API calls per second
4. **Response Time**: 95th percentile latency

### Custom Queries
```promql
# Face recognition success rate
rate(face_recognition_success_total[5m]) / rate(face_recognition_requests_total[5m]) * 100

# Average response time
histogram_quantile(0.95, rate(face_recognition_duration_seconds_bucket[5m]))
```

## 🔍 Logging

### Structured Logs
```python
# Example log entry
{
  "timestamp": "2025-01-02T12:00:00Z",
  "level": "INFO",
  "message": "Face recognition successful",
  "user_id": "1004686",
  "door_id": "19456",
  "action": "recognize_face",
  "success": true,
  "duration_ms": 150
}
```

### Log Queries in Kibana
```json
// Find all face recognition attempts
{
  "query": {
    "bool": {
      "must": [
        {"term": {"action": "recognize_face"}},
        {"range": {"timestamp": {"gte": "now-1h"}}}
      ]
    }
  }
}
```

## 🚨 Alerts

### Prometheus Alert Rules
```yaml
groups:
- name: ftl-gym-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(face_recognition_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
```

## 🔧 Maintenance

### Backup
```bash
# Database backup
docker-compose exec postgres pg_dump -U postgres ftl_gym > backup.sql

# Elasticsearch backup
curl -X POST "localhost:9200/_snapshot/backup/snapshot_1"
```

### Scaling
```bash
# Scale application instances
docker-compose up -d --scale ftl-gym-app=3

# Scale with load balancer
docker-compose up -d nginx
```

### Updates
```bash
# Update all services
docker-compose pull
docker-compose up -d

# Update specific service
docker-compose pull ftl-gym-app
docker-compose up -d ftl-gym-app
```

## 🐛 Troubleshooting

### Common Issues
1. **Out of Memory**: Increase Docker memory limit
2. **Port Conflicts**: Check if ports are already in use
3. **Database Connection**: Verify PostgreSQL is running
4. **Elasticsearch**: Check cluster health

### Debug Commands
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs ftl-gym-app

# Check resource usage
docker stats

# Access container shell
docker-compose exec ftl-gym-app bash
```

## 📊 Performance Tuning

### Resource Allocation
```yaml
# docker-compose.yml
services:
  ftl-gym-app:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Database Optimization
```sql
-- Create indexes for better performance
CREATE INDEX idx_face_embeddings_user_id ON face_embeddings(user_id);
CREATE INDEX idx_sessions_token ON sessions(token);
```

## 🔒 Security

### Network Security
- All services run in isolated network
- Nginx provides rate limiting
- HTTPS termination (configure SSL certificates)

### Data Protection
- Encrypted database connections
- Secure session management
- Face data encryption at rest

## 📝 Logs

### Application Logs
```bash
# View application logs
docker-compose logs -f ftl-gym-app

# View specific service logs
docker-compose logs -f elasticsearch
```

### System Logs
```bash
# View all logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f
```

## 🚀 Production Deployment

### GCloud Deployment
```bash
# Build and push to GCR
docker build -t gcr.io/your-project/ftl-gym .
docker push gcr.io/your-project/ftl-gym

# Deploy to GKE
kubectl apply -f k8s/
```

### Environment Variables
```bash
# Production environment
export FLASK_ENV=production
export DATABASE_URL=postgresql://user:pass@db:5432/ftl_gym
export REDIS_URL=redis://redis:6379
```

## 📞 Support

### Monitoring
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601

### Health Checks
```bash
# Application health
curl http://localhost:8080/health

# Database health
docker-compose exec postgres pg_isready

# Redis health
docker-compose exec redis redis-cli ping
```

---

**FTL GYM Face Recognition System**  
*Secure biometric authentication and identity verification*
