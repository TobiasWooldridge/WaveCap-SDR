import { Component, ReactNode } from 'react';
import Flex from './primitives/Flex.react';
import Button from './primitives/Button.react';
import { AlertTriangle } from 'lucide-react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({
      error,
      errorInfo,
    });
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="container py-5">
          <div className="card shadow">
            <div className="card-body">
              <Flex direction="column" align="center" gap={3} className="py-4">
                <AlertTriangle size={64} className="text-danger" />
                <h2 className="h4">Something went wrong</h2>
                <p className="text-muted text-center">
                  The application encountered an unexpected error. Don't worry, your data is safe.
                </p>

                {this.state.error && (
                  <details className="w-100">
                    <summary className="btn btn-sm btn-outline-secondary">Show error details</summary>
                    <div className="mt-3 p-3 bg-light border rounded">
                      <p className="small mb-2">
                        <strong>Error:</strong> {this.state.error.toString()}
                      </p>
                      {this.state.errorInfo && (
                        <pre className="small mb-0" style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                          {this.state.errorInfo.componentStack}
                        </pre>
                      )}
                    </div>
                  </details>
                )}

                <Flex gap={2}>
                  <Button use="primary" onClick={this.handleReset}>
                    Try Again
                  </Button>
                  <Button use="secondary" onClick={() => window.location.reload()}>
                    Reload Page
                  </Button>
                </Flex>
              </Flex>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
