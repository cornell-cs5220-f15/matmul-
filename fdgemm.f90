SUBROUTINE DGEMM(A,lda,lta,B,ldb,ltb,C,ldc,ltc)

IMPLICIT NONE

INTEGER, INTENT(IN) :: lda, lta, ldb, ltb, ldc, ltc

REAL*8 , INTENT(IN) :: a(1:lda, 1:lta), b(1:ldb, 1:ltb)
REAL*8 , INTENT(INOUT):: c(1:ldc, 1:ltc)

INTEGER :: j, k

!
!----------------------------------------------------------------------
!write(*,*) 'got here'
!c = 0.0

do k = 1,ltb
    do j = 1,lta
        c(:, k) = c(:, k) + a(:, j)*b(j, k)
    end do
end do

END SUBROUTINE DGEMM

