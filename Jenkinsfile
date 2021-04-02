pipeline {
    agent {
        docker { 
		image 'docker.pkg.github.com/benjamintrapani/fungpu/fungpu_built:latest'
		args '--gpus all'
	}
    }
    stages {
        stage('build') {
            steps {
                sh "bash build.sh"
            }
        }
        stage('test') {
            steps {
                sh "bash test.sh"
            }
        }
    }
}
