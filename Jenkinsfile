pipeline {
    agent any
    environment {
        container_name = "c_${BUILD_ID}_${JENKINS_NODE_COOKIE}"
        image_tag = "sal_v4.0.0_salobj_v5.0.0"
        user_ci = credentials('lsst-io')
    }

    stages {
        stage("Pull docker image") {
            steps {
                script {
                    sh """
                    docker pull lsstts/develop-env:\${image_tag}
                    """
                }
            }
        }
        stage("Prepare Workspace") {
            steps {
                script {
                    sh """
                    chmod -R a+rw \${WORKSPACE} || echo "Failed to set workspace mode"
                    """
                }
            }
        }
        stage("Run container") {
            steps {
                script {
                    sh """
                    container=\$(docker run -v \${WORKSPACE}:/home/saluser/repo/ -td --rm --name \${container_name} -e LTD_USERNAME=\${user_ci_USR} -e LTD_PASSWORD=\${user_ci_PSW} lsstts/develop-env:\${image_tag})
                    """
                }
            }
        }
        stage("Build IDL Files") {
            steps {
                script {
                    sh """
                    docker exec -u saluser \${container_name} sh -c \"source ~/.setup.sh && source ~/.bashrc && make_idl_files.py ATAOS ATMCS ATPneumatics ATHexapod ATCamera ATPtg\"
                    """
                }
            }
        }
        stage("Running tests") {
            steps {
                script {
                    sh """
                    docker exec -u saluser \${container_name} sh -c \"source ~/.setup.sh && cd repo && eups declare -r . -t saluser && setup ts_ataos -t saluser && scons\"
                    """
                }
            }
        }
    }
    post {
        always {
            // The path of xml needed by JUnit is relative to
            // the workspace.
            junit 'tests/.tests/*.xml'

            // Publish the HTML report
            publishHTML (target: [
                allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: true,
                reportDir: 'tests/.tests/pytest-ts_ataos.xml-htmlcov/',
                reportFiles: 'index.html',
                reportName: "Coverage Report"
              ])
        }
        cleanup {
            sh """
                docker exec -u root --privileged \${container_name} sh -c \"chmod -R a+rw /home/saluser/repo/ \"
                docker stop \${container_name}
            """
            deleteDir()
        }
    }
}
