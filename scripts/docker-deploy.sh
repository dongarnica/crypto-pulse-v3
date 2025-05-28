#!/bin/bash
# Docker deployment and management scripts for Crypto Pulse V3

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="crypto-pulse-v3"
DOCKER_COMPOSE_FILE="docker-compose.yml"
DOCKER_COMPOSE_PROD_FILE="docker-compose.prod.yml"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are installed
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "Dependencies check passed"
}

# Build all images
build_images() {
    log_info "Building Docker images..."
    docker-compose -f $DOCKER_COMPOSE_FILE build --no-cache
    log_success "Images built successfully"
}

# Development environment
start_dev() {
    log_info "Starting development environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        log_warning ".env file not found. Creating from example..."
        cp .env.example .env
        log_warning "Please edit .env file with your configuration before continuing."
        exit 1
    fi
    
    # Start services
    docker-compose -f $DOCKER_COMPOSE_FILE up -d
    
    log_success "Development environment started"
    log_info "Services available at:"
    log_info "  - API Server: http://localhost:8000"
    log_info "  - Database: localhost:5432"
    log_info "  - Redis: localhost:6379"
    
    # Show logs
    log_info "Following logs (Ctrl+C to stop)..."
    docker-compose -f $DOCKER_COMPOSE_FILE logs -f
}

# Production environment
start_prod() {
    log_info "Starting production environment..."
    
    # Check for production env file
    if [ ! -f .env.production ]; then
        log_error ".env.production file not found. Please create it from .env.production.example"
        exit 1
    fi
    
    # Start production services
    docker-compose -f $DOCKER_COMPOSE_PROD_FILE --env-file .env.production up -d
    
    log_success "Production environment started"
    log_info "Services available at:"
    log_info "  - API Server: http://localhost:8000"
    log_info "  - Monitoring: http://localhost:3000"
}

# Stop services
stop_services() {
    log_info "Stopping services..."
    
    if [ "$1" = "prod" ]; then
        docker-compose -f $DOCKER_COMPOSE_PROD_FILE down
    else
        docker-compose -f $DOCKER_COMPOSE_FILE down
    fi
    
    log_success "Services stopped"
}

# Clean up everything
cleanup() {
    log_info "Cleaning up Docker resources..."
    
    # Stop and remove containers
    docker-compose -f $DOCKER_COMPOSE_FILE down --volumes --remove-orphans 2>/dev/null || true
    docker-compose -f $DOCKER_COMPOSE_PROD_FILE down --volumes --remove-orphans 2>/dev/null || true
    
    # Remove project images
    docker images | grep $PROJECT_NAME | awk '{print $3}' | xargs docker rmi -f 2>/dev/null || true
    
    # Clean up unused resources
    docker system prune -f
    
    log_success "Cleanup completed"
}

# Show service status
status() {
    log_info "Service status:"
    
    if [ "$1" = "prod" ]; then
        docker-compose -f $DOCKER_COMPOSE_PROD_FILE ps
    else
        docker-compose -f $DOCKER_COMPOSE_FILE ps
    fi
}

# Show logs
show_logs() {
    log_info "Showing logs..."
    
    if [ "$1" = "prod" ]; then
        docker-compose -f $DOCKER_COMPOSE_PROD_FILE logs -f "${2:-}"
    else
        docker-compose -f $DOCKER_COMPOSE_FILE logs -f "${2:-}"
    fi
}

# Run backtesting
run_backtest() {
    log_info "Running backtesting container..."
    
    docker-compose -f $DOCKER_COMPOSE_FILE run --rm backtesting python working_advanced_cli.py demo
    
    log_success "Backtesting completed"
}

# Database operations
db_init() {
    log_info "Initializing database..."
    
    if [ "$1" = "prod" ]; then
        docker-compose -f $DOCKER_COMPOSE_PROD_FILE run --rm db-migrate
    else
        docker-compose -f $DOCKER_COMPOSE_FILE run --rm db-migrate
    fi
    
    log_success "Database initialized"
}

# Backup database
db_backup() {
    local backup_file="backup_$(date +%Y%m%d_%H%M%S).sql"
    log_info "Creating database backup: $backup_file"
    
    if [ "$1" = "prod" ]; then
        docker-compose -f $DOCKER_COMPOSE_PROD_FILE exec postgres pg_dump -U cryptopulse crypto_pulse_v3 > "$backup_file"
    else
        docker-compose -f $DOCKER_COMPOSE_FILE exec postgres pg_dump -U postgres crypto_pulse_v3 > "$backup_file"
    fi
    
    log_success "Database backup created: $backup_file"
}

# Show help
show_help() {
    echo "Crypto Pulse V3 Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build              Build Docker images"
    echo "  dev                Start development environment"
    echo "  prod               Start production environment"
    echo "  stop [prod]        Stop services (add 'prod' for production)"
    echo "  status [prod]      Show service status"
    echo "  logs [prod] [service]  Show logs"
    echo "  backtest           Run backtesting"
    echo "  db-init [prod]     Initialize database"
    echo "  db-backup [prod]   Backup database"
    echo "  cleanup            Clean up all Docker resources"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 dev             # Start development environment"
    echo "  $0 prod            # Start production environment"
    echo "  $0 logs prod app   # Show production app logs"
    echo "  $0 stop prod       # Stop production environment"
}

# Main script logic
main() {
    case "${1:-}" in
        build)
            check_dependencies
            build_images
            ;;
        dev)
            check_dependencies
            start_dev
            ;;
        prod)
            check_dependencies
            start_prod
            ;;
        stop)
            stop_services "$2"
            ;;
        status)
            status "$2"
            ;;
        logs)
            show_logs "$2" "$3"
            ;;
        backtest)
            run_backtest
            ;;
        db-init)
            db_init "$2"
            ;;
        db-backup)
            db_backup "$2"
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: ${1:-}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
