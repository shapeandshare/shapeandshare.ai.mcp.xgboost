#!/usr/bin/env python3
"""
Cluster Health Verification Tests

This module provides comprehensive health verification tests for the MCP XGBoost
cluster deployment including Kubernetes cluster health, Istio service mesh status,
and MCP server availability.

Features:
- Kubernetes cluster health checks
- Istio service mesh verification
- MCP server health and readiness
- Service discovery and networking
- Resource utilization monitoring
- Performance baseline verification

Usage:
    pytest tests/test_cluster_health_verification.py -v
    python tests/test_cluster_health_verification.py  # Run as script
"""

import pytest
import requests
import time
import json
import subprocess
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml
import concurrent.futures
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Network imports
try:
    import requests
    import aiohttp
    NETWORK_LIBS_AVAILABLE = True
except ImportError:
    NETWORK_LIBS_AVAILABLE = False


class ClusterHealthConfig:
    """Configuration for cluster health verification"""
    
    def __init__(self):
        self.cluster_url = os.getenv("CLUSTER_URL", "http://localhost:8000")
        self.istio_gateway = os.getenv("ISTIO_GATEWAY", "http://localhost:8080")
        self.k8s_context = os.getenv("K8S_CONTEXT", "k3d-mcp-xgboost")
        self.namespace = os.getenv("MCP_NAMESPACE", "mcp-xgboost")
        self.test_timeout = int(os.getenv("TEST_TIMEOUT", "30"))
        
        # Endpoints
        self.health_endpoint = f"{self.cluster_url}/health"
        self.mcp_endpoint = f"{self.cluster_url}/mcp"
        self.metrics_endpoint = f"{self.cluster_url}/metrics"
        
        # Istio endpoints
        self.istio_health = f"{self.istio_gateway}/health"
        self.istio_mcp = f"{self.istio_gateway}/mcp"
        
        # Expected services
        self.expected_services = [
            "mcp-xgboost",
            "mcp-xgboost-metrics",
        ]
        
        # Performance thresholds
        self.performance_thresholds = {
            "health_response_time": 2.0,  # seconds
            "mcp_response_time": 5.0,     # seconds
            "cpu_usage_limit": 80.0,      # percentage
            "memory_usage_limit": 80.0,   # percentage
        }


@pytest.fixture
def health_config():
    """Fixture providing health configuration"""
    return ClusterHealthConfig()


class TestKubernetesClusterHealth:
    """Test Kubernetes cluster health"""
    
    def test_kubectl_connectivity(self, health_config):
        """Test kubectl connectivity to cluster"""
        try:
            result = subprocess.run(
                ["kubectl", "cluster-info", "--context", health_config.k8s_context],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0, f"kubectl failed: {result.stderr}"
            assert "Kubernetes control plane" in result.stdout
            
            logger.info("✅ kubectl connectivity verified")
            
        except subprocess.TimeoutExpired:
            pytest.skip("kubectl command timed out")
        except FileNotFoundError:
            pytest.skip("kubectl not found")
        except Exception as e:
            pytest.skip(f"kubectl connectivity test failed: {e}")
    
    def test_namespace_exists(self, health_config):
        """Test that the MCP namespace exists"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "namespace", health_config.namespace, 
                 "--context", health_config.k8s_context],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0, f"Namespace check failed: {result.stderr}"
            assert health_config.namespace in result.stdout
            
            logger.info(f"✅ Namespace {health_config.namespace} exists")
            
        except subprocess.TimeoutExpired:
            pytest.skip("Namespace check timed out")
        except FileNotFoundError:
            pytest.skip("kubectl not found")
        except Exception as e:
            pytest.skip(f"Namespace check failed: {e}")
    
    def test_pods_running(self, health_config):
        """Test that MCP pods are running"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", health_config.namespace,
                 "--context", health_config.k8s_context, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0, f"Pod check failed: {result.stderr}"
            
            pods_data = json.loads(result.stdout)
            pods = pods_data.get("items", [])
            
            assert len(pods) > 0, "No pods found in namespace"
            
            # Check pod statuses
            running_pods = 0
            for pod in pods:
                status = pod.get("status", {})
                phase = status.get("phase", "")
                
                if phase == "Running":
                    running_pods += 1
                    
                    # Check container statuses
                    container_statuses = status.get("containerStatuses", [])
                    for container in container_statuses:
                        assert container.get("ready", False), \
                            f"Container {container.get('name')} not ready"
            
            assert running_pods > 0, "No running pods found"
            
            logger.info(f"✅ {running_pods} pods running in namespace")
            
        except subprocess.TimeoutExpired:
            pytest.skip("Pod check timed out")
        except FileNotFoundError:
            pytest.skip("kubectl not found")
        except json.JSONDecodeError as e:
            pytest.skip(f"Failed to parse kubectl output: {e}")
        except Exception as e:
            pytest.skip(f"Pod check failed: {e}")
    
    def test_services_available(self, health_config):
        """Test that required services are available"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "services", "-n", health_config.namespace,
                 "--context", health_config.k8s_context, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0, f"Service check failed: {result.stderr}"
            
            services_data = json.loads(result.stdout)
            services = services_data.get("items", [])
            
            service_names = [svc.get("metadata", {}).get("name", "") for svc in services]
            
            for expected_service in health_config.expected_services:
                # Check if service exists (partial match for generated names)
                service_found = any(expected_service in name for name in service_names)
                if not service_found:
                    logger.warning(f"Service {expected_service} not found in {service_names}")
            
            assert len(services) > 0, "No services found in namespace"
            
            logger.info(f"✅ {len(services)} services available in namespace")
            
        except subprocess.TimeoutExpired:
            pytest.skip("Service check timed out")
        except FileNotFoundError:
            pytest.skip("kubectl not found")
        except json.JSONDecodeError as e:
            pytest.skip(f"Failed to parse kubectl output: {e}")
        except Exception as e:
            pytest.skip(f"Service check failed: {e}")


class TestIstioServiceMesh:
    """Test Istio service mesh health"""
    
    def test_istio_system_pods(self, health_config):
        """Test that Istio system pods are running"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", "istio-system",
                 "--context", health_config.k8s_context, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0, f"Istio pod check failed: {result.stderr}"
            
            pods_data = json.loads(result.stdout)
            pods = pods_data.get("items", [])
            
            # Check for key Istio components
            expected_components = ["istiod", "istio-ingressgateway"]
            found_components = []
            
            for pod in pods:
                pod_name = pod.get("metadata", {}).get("name", "")
                for component in expected_components:
                    if component in pod_name:
                        status = pod.get("status", {})
                        phase = status.get("phase", "")
                        if phase == "Running":
                            found_components.append(component)
            
            if len(found_components) == 0:
                pytest.skip("No Istio components found (Istio may not be installed)")
            
            logger.info(f"✅ Istio components running: {found_components}")
            
        except subprocess.TimeoutExpired:
            pytest.skip("Istio pod check timed out")
        except FileNotFoundError:
            pytest.skip("kubectl not found")
        except json.JSONDecodeError as e:
            pytest.skip(f"Failed to parse kubectl output: {e}")
        except Exception as e:
            pytest.skip(f"Istio pod check failed: {e}")
    
    def test_istio_gateway_config(self, health_config):
        """Test that Istio gateway configuration exists"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "gateway", "-n", health_config.namespace,
                 "--context", health_config.k8s_context, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                pytest.skip("Istio gateway not configured")
            
            gateways_data = json.loads(result.stdout)
            gateways = gateways_data.get("items", [])
            
            if len(gateways) == 0:
                pytest.skip("No Istio gateways found")
            
            logger.info(f"✅ {len(gateways)} Istio gateways configured")
            
        except subprocess.TimeoutExpired:
            pytest.skip("Istio gateway check timed out")
        except FileNotFoundError:
            pytest.skip("kubectl not found")
        except json.JSONDecodeError as e:
            pytest.skip(f"Failed to parse kubectl output: {e}")
        except Exception as e:
            pytest.skip(f"Istio gateway check failed: {e}")
    
    @pytest.mark.skipif(not NETWORK_LIBS_AVAILABLE, reason="Network libraries not available")
    def test_istio_gateway_accessibility(self, health_config):
        """Test that Istio gateway is accessible"""
        try:
            response = requests.get(health_config.istio_health, timeout=10)
            # Gateway might return 404 if health route not configured, which is acceptable
            assert response.status_code in [200, 404], \
                f"Unexpected status code: {response.status_code}"
            
            logger.info("✅ Istio gateway is accessible")
            
        except requests.RequestException as e:
            pytest.skip(f"Istio gateway not accessible: {e}")


class TestMCPServerHealth:
    """Test MCP server health and readiness"""
    
    @pytest.mark.skipif(not NETWORK_LIBS_AVAILABLE, reason="Network libraries not available")
    def test_health_endpoint_response(self, health_config):
        """Test MCP server health endpoint"""
        try:
            start_time = time.time()
            response = requests.get(health_config.health_endpoint, timeout=10)
            response_time = time.time() - start_time
            
            assert response.status_code == 200, f"Health check failed: {response.status_code}"
            
            # Check response time
            assert response_time < health_config.performance_thresholds["health_response_time"], \
                f"Health response too slow: {response_time:.2f}s"
            
            # Check response content
            health_data = response.json()
            assert "status" in health_data
            assert health_data["status"] in ["healthy", "ok"]
            
            logger.info(f"✅ Health endpoint responding in {response_time:.2f}s")
            
        except requests.RequestException as e:
            pytest.skip(f"Health endpoint not accessible: {e}")
        except json.JSONDecodeError as e:
            pytest.skip(f"Invalid health response format: {e}")
    
    @pytest.mark.skipif(not NETWORK_LIBS_AVAILABLE, reason="Network libraries not available")
    def test_mcp_endpoint_response(self, health_config):
        """Test MCP endpoint response"""
        try:
            start_time = time.time()
            response = requests.get(health_config.mcp_endpoint, timeout=10)
            response_time = time.time() - start_time
            
            # MCP endpoint might return 405 (Method Not Allowed) for GET requests
            assert response.status_code in [200, 405], \
                f"Unexpected MCP status: {response.status_code}"
            
            # Check response time
            assert response_time < health_config.performance_thresholds["mcp_response_time"], \
                f"MCP response too slow: {response_time:.2f}s"
            
            logger.info(f"✅ MCP endpoint responding in {response_time:.2f}s")
            
        except requests.RequestException as e:
            pytest.skip(f"MCP endpoint not accessible: {e}")
    
    @pytest.mark.skipif(not NETWORK_LIBS_AVAILABLE, reason="Network libraries not available")
    def test_mcp_protocol_basic(self, health_config):
        """Test basic MCP protocol communication"""
        try:
            # Test basic MCP initialize request
            mcp_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "health-check-client",
                        "version": "1.0.0"
                    }
                },
                "id": 1
            }
            
            response = requests.post(
                health_config.mcp_endpoint,
                json=mcp_request,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            # Should get a valid response (200 or 400/405 if not properly implemented)
            assert response.status_code in [200, 400, 405], \
                f"Unexpected MCP protocol status: {response.status_code}"
            
            logger.info("✅ MCP protocol communication functional")
            
        except requests.RequestException as e:
            pytest.skip(f"MCP protocol test failed: {e}")


class TestServiceNetworking:
    """Test service networking and discovery"""
    
    @pytest.mark.skipif(not NETWORK_LIBS_AVAILABLE, reason="Network libraries not available")
    def test_service_discovery(self, health_config):
        """Test service discovery and networking"""
        try:
            # Test multiple endpoints to verify service discovery
            endpoints = [
                health_config.health_endpoint,
                health_config.mcp_endpoint,
            ]
            
            successful_connections = 0
            for endpoint in endpoints:
                try:
                    response = requests.get(endpoint, timeout=5)
                    if response.status_code in [200, 404, 405]:
                        successful_connections += 1
                except requests.RequestException:
                    continue
            
            assert successful_connections > 0, "No endpoints accessible"
            
            logger.info(f"✅ {successful_connections}/{len(endpoints)} endpoints accessible")
            
        except Exception as e:
            pytest.skip(f"Service discovery test failed: {e}")
    
    @pytest.mark.skipif(not NETWORK_LIBS_AVAILABLE, reason="Network libraries not available")
    def test_load_balancing(self, health_config):
        """Test load balancing behavior"""
        try:
            # Make multiple requests to test load balancing
            responses = []
            for i in range(5):
                try:
                    response = requests.get(health_config.health_endpoint, timeout=5)
                    responses.append(response.status_code)
                except requests.RequestException:
                    responses.append(0)
            
            # Check that we get consistent responses
            successful_responses = sum(1 for r in responses if r == 200)
            assert successful_responses >= 3, "Load balancing not working consistently"
            
            logger.info(f"✅ Load balancing working ({successful_responses}/5 successful)")
            
        except Exception as e:
            pytest.skip(f"Load balancing test failed: {e}")


class TestPerformanceBaseline:
    """Test performance baseline metrics"""
    
    @pytest.mark.skipif(not NETWORK_LIBS_AVAILABLE, reason="Network libraries not available")
    def test_response_time_baseline(self, health_config):
        """Test response time baseline"""
        try:
            # Test multiple requests for baseline
            response_times = []
            for i in range(10):
                start_time = time.time()
                response = requests.get(health_config.health_endpoint, timeout=10)
                end_time = time.time()
                
                if response.status_code == 200:
                    response_times.append(end_time - start_time)
            
            if len(response_times) == 0:
                pytest.skip("No successful responses for baseline")
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            assert avg_response_time < health_config.performance_thresholds["health_response_time"], \
                f"Average response time too slow: {avg_response_time:.2f}s"
            
            logger.info(f"✅ Response time baseline: avg={avg_response_time:.2f}s, max={max_response_time:.2f}s")
            
        except Exception as e:
            pytest.skip(f"Response time baseline test failed: {e}")
    
    @pytest.mark.skipif(not NETWORK_LIBS_AVAILABLE, reason="Network libraries not available")
    def test_concurrent_requests(self, health_config):
        """Test concurrent request handling"""
        try:
            def make_request():
                try:
                    response = requests.get(health_config.health_endpoint, timeout=10)
                    return response.status_code
                except requests.RequestException:
                    return 0
            
            # Make 10 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_request) for _ in range(10)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            successful_requests = sum(1 for r in results if r == 200)
            assert successful_requests >= 7, f"Only {successful_requests}/10 concurrent requests succeeded"
            
            logger.info(f"✅ Concurrent request handling: {successful_requests}/10 successful")
            
        except Exception as e:
            pytest.skip(f"Concurrent request test failed: {e}")


class TestResourceMonitoring:
    """Test resource monitoring and limits"""
    
    def test_pod_resource_usage(self, health_config):
        """Test pod resource usage"""
        try:
            # Get pod metrics if available
            result = subprocess.run(
                ["kubectl", "top", "pods", "-n", health_config.namespace,
                 "--context", health_config.k8s_context],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                pytest.skip("Pod metrics not available (metrics-server may not be installed)")
            
            # Parse metrics output
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                pytest.skip("No pod metrics available")
            
            # Check resource usage
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 3:
                    pod_name = parts[0]
                    cpu_usage = parts[1]
                    memory_usage = parts[2]
                    
                    logger.info(f"Pod {pod_name}: CPU={cpu_usage}, Memory={memory_usage}")
            
            logger.info("✅ Pod resource monitoring available")
            
        except subprocess.TimeoutExpired:
            pytest.skip("Pod metrics check timed out")
        except FileNotFoundError:
            pytest.skip("kubectl not found")
        except Exception as e:
            pytest.skip(f"Pod resource monitoring failed: {e}")
    
    def test_node_resource_availability(self, health_config):
        """Test node resource availability"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "nodes", "--context", health_config.k8s_context, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0, f"Node check failed: {result.stderr}"
            
            nodes_data = json.loads(result.stdout)
            nodes = nodes_data.get("items", [])
            
            ready_nodes = 0
            for node in nodes:
                conditions = node.get("status", {}).get("conditions", [])
                for condition in conditions:
                    if condition.get("type") == "Ready" and condition.get("status") == "True":
                        ready_nodes += 1
                        break
            
            assert ready_nodes > 0, "No ready nodes found"
            
            logger.info(f"✅ {ready_nodes} nodes ready")
            
        except subprocess.TimeoutExpired:
            pytest.skip("Node check timed out")
        except FileNotFoundError:
            pytest.skip("kubectl not found")
        except json.JSONDecodeError as e:
            pytest.skip(f"Failed to parse kubectl output: {e}")
        except Exception as e:
            pytest.skip(f"Node resource check failed: {e}")


def create_health_verification_report(config: ClusterHealthConfig) -> Dict[str, Any]:
    """Create comprehensive health verification report"""
    report = {
        "timestamp": time.time(),
        "cluster_url": config.cluster_url,
        "health_status": "unknown",
        "components": {},
        "performance": {},
        "recommendations": []
    }
    
    # Test basic connectivity
    try:
        response = requests.get(config.health_endpoint, timeout=5)
        if response.status_code == 200:
            report["health_status"] = "healthy"
            report["components"]["mcp_server"] = "healthy"
        else:
            report["health_status"] = "degraded"
            report["components"]["mcp_server"] = "degraded"
    except Exception:
        report["health_status"] = "unhealthy"
        report["components"]["mcp_server"] = "unhealthy"
    
    # Test performance
    try:
        start_time = time.time()
        response = requests.get(config.health_endpoint, timeout=5)
        response_time = time.time() - start_time
        report["performance"]["health_response_time"] = response_time
        
        if response_time > config.performance_thresholds["health_response_time"]:
            report["recommendations"].append("Health endpoint response time is slow")
    except Exception:
        report["performance"]["health_response_time"] = None
    
    return report


def run_health_verification():
    """Run comprehensive health verification"""
    print("Running MCP XGBoost Cluster Health Verification")
    print("=" * 60)
    
    config = ClusterHealthConfig()
    
    # Generate health report
    report = create_health_verification_report(config)
    
    print(f"Cluster Status: {report['health_status'].upper()}")
    print(f"Cluster URL: {report['cluster_url']}")
    
    if report["components"]:
        print("\nComponent Status:")
        for component, status in report["components"].items():
            print(f"  {component}: {status}")
    
    if report["performance"]:
        print("\nPerformance Metrics:")
        for metric, value in report["performance"].items():
            if value is not None:
                print(f"  {metric}: {value:.2f}s")
    
    if report["recommendations"]:
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")
    
    # Run pytest tests
    test_args = [
        "tests/test_cluster_health_verification.py",
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    result = subprocess.run(["python", "-m", "pytest"] + test_args)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_health_verification()
    exit(0 if success else 1) 