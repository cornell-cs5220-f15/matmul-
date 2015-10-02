SUBROUTINE DGEMM(A,lda,B,ldb,C,ldc)

IMPLICIT NONE

INTEGER, INTENT(IN) :: lda, ldb, ldc

REAL(15, 307), INTENT(IN) :: a(lda, *), b(ldb, *)
REAL(15, 307), INTENT(INOUT):: c(ldc, *)

INTEGER :: j, k

INTEGER lta, ltb
!
!----------------------------------------------------------------------

lda = ubound(a, 1)
ldb = ubound(b, 1)
lta = ubound(a, 2)
ltb = ubound(b, 2)

c = 0.0

do k = 1,ltb
    do j = 1,lda
        c(:, j) = c(:, j) + a(j, k)*b(:, k)
    end do
end do



END SUBROUTINE DGEMM

